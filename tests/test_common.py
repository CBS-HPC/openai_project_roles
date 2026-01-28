from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from openai_project_roles.app import common


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_subtract_months_clamps_day() -> None:
    dt = datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc)
    out = common.subtract_months(dt, 1)
    assert out.year == 2026
    assert out.month == 2
    assert out.day == 28


def test_to_unix_seconds_naive_assumes_utc() -> None:
    dt = datetime(2026, 1, 1, 0, 0, 0)
    assert common.to_unix_seconds(dt) == int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())


def test_load_and_save_project_budgets(tmp_path: Path) -> None:
    path = tmp_path / "budgets.yaml"
    common.save_project_budgets({"p1": 10, "p2": 20.5}, budget_file=path)

    loaded = common.load_project_budgets(path)
    assert loaded == {"p1": 10.0, "p2": 20.5}

    # malformed values are ignored
    path.write_text("budgets:\n  p1: null\n  p2: abc\n  p3: 7\n", encoding="utf-8")
    loaded = common.load_project_budgets(path)
    assert loaded == {"p3": 7.0}


def test_load_roles_config_local(tmp_path: Path) -> None:
    path = tmp_path / "roles.yaml"
    path.write_text("roles:\n  - role_name: reader\n    permissions: [read]\n", encoding="utf-8")
    roles = common.load_roles_config(str(path))
    assert roles == [{"role_name": "reader", "permissions": ["read"]}]


def test_download_roles_config(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "roles.yaml"

    def _fake_get(url: str, timeout: int = 10) -> _FakeResponse:
        return _FakeResponse({"ok": True}, status_code=200)

    def _fake_get_text(url: str, timeout: int = 10) -> _FakeResponse:
        resp = _FakeResponse({}, status_code=200)
        resp.text = "roles:\n  - role_name: admin\n    permissions: [all]\n"
        return resp

    # monkeypatch requests.get to return a text payload
    monkeypatch.setattr(common.requests, "get", _fake_get_text)

    assert common.download_roles_config("http://example.com/roles.yaml", str(path)) is True
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["roles"][0]["role_name"] == "admin"


def test_fetch_costs_by_project_parses_usd(monkeypatch) -> None:
    payload = {
        "data": [
            {
                "start_time": int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()),
                "results": [
                    {"project_id": "p1", "amount": {"value": 2.5, "currency": "usd"}},
                    {"project_id": "p1", "amount": {"value": 1.0, "currency": "USD"}},
                    {"project_id": "p2", "amount": {"value": 3.0, "currency": "eur"}},
                ],
            }
        ],
        "has_more": False,
    }

    def _fake_get(url, headers=None, params=None, timeout=60):
        return _FakeResponse(payload)

    monkeypatch.setattr(common.requests, "get", _fake_get)

    totals, daily = common.fetch_costs_by_project(
        api_key="sk-test",
        start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2026, 1, 2, tzinfo=timezone.utc),
        project_id=None,
    )

    assert totals["p1"] == 3.5
    assert "p2" not in totals
    assert daily["p1"]["2026-01-01"] == 3.5
