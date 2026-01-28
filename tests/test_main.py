from __future__ import annotations

import sys

from tests.conftest import DummyStreamlit

from openai_project_roles.app import main


def test_parse_script_args(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py", "--roles-config-path", "r.yaml", "--budget-path", "b.yaml", "--usage-path", "u.csv"])
    args = main._parse_script_args()
    assert args.roles_config_path == "r.yaml"
    assert args.budget_path == "b.yaml"
    assert args.usage_path == "u.csv"


def test_main_calls_tabs_and_render(monkeypatch) -> None:
    dummy = DummyStreamlit()

    monkeypatch.setattr(main, "st", dummy)

    called = {"creds": False, "roles": False, "usage": False}

    def _creds():
        called["creds"] = True

    def _roles(*args, **kwargs):
        called["roles"] = True

    def _usage(*args, **kwargs):
        called["usage"] = True

    monkeypatch.setattr(main, "render_credentials_section", _creds)
    monkeypatch.setattr(main, "render_roles_tab", _roles)
    monkeypatch.setattr(main, "render_usage_tab", _usage)

    main.main("roles.yaml", "budgets.yaml", "usage.csv")

    assert called["creds"] is True
    assert called["roles"] is True
    assert called["usage"] is True
    assert "set_page_config" in dummy.calls
    assert "tabs" in dummy.calls
