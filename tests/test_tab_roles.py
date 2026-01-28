from __future__ import annotations

from tests.conftest import DummyStreamlit

from openai_project_roles.app import tab_roles


def test_render_roles_tab_when_roles_missing(monkeypatch) -> None:
    dummy = DummyStreamlit()
    dummy.session_state["api_key"] = ""
    dummy.session_state["project_id"] = ""

    monkeypatch.setattr(tab_roles, "st", dummy)
    monkeypatch.setattr(tab_roles, "load_roles_config", lambda roles_config_path: [])
    monkeypatch.setattr(tab_roles, "download_roles_config", lambda *args, **kwargs: False)
    monkeypatch.setattr(tab_roles, "browse_for_yaml", lambda *args, **kwargs: None)

    tab_roles.render_roles_tab(roles_config_path="missing.yaml")

    assert "warning" in dummy.calls
    assert "info" in dummy.calls


def test_render_roles_tab_with_roles_no_create(monkeypatch) -> None:
    dummy = DummyStreamlit()
    dummy.session_state["api_key"] = "sk-test"
    dummy.session_state["project_id"] = "proj_123"

    monkeypatch.setattr(tab_roles, "st", dummy)
    monkeypatch.setattr(
        tab_roles,
        "load_roles_config",
        lambda roles_config_path: [
            {"role_name": "reader", "permissions": ["read"], "description": "desc"}
        ],
    )
    monkeypatch.setattr(tab_roles, "browse_for_yaml", lambda *args, **kwargs: None)

    tab_roles.render_roles_tab(roles_config_path="roles.yaml")

    assert "checkbox" in dummy.calls
    assert any(call.startswith("button:") for call in dummy.calls)
