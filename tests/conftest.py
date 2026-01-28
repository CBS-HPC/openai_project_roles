from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

from openai_project_roles.app import common as _common

APP_DIR = Path(__file__).resolve().parents[1] / "openai_project_roles" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Allow app modules that use "from common import ..." to resolve in tests.
sys.modules.setdefault("common", _common)


class _DummyCtx:
    def __enter__(self) -> "_DummyCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        return self.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.calls: List[str] = []
        self._button_responses: Dict[str, bool] = {}

    def set_button_response(self, label: str, value: bool) -> None:
        self._button_responses[label] = value

    def title(self, *args, **kwargs) -> None:
        self.calls.append("title")

    def markdown(self, *args, **kwargs) -> None:
        self.calls.append("markdown")

    def divider(self, *args, **kwargs) -> None:
        self.calls.append("divider")

    def subheader(self, *args, **kwargs) -> None:
        self.calls.append("subheader")

    def caption(self, *args, **kwargs) -> None:
        self.calls.append("caption")

    def code(self, *args, **kwargs) -> None:
        self.calls.append("code")

    def warning(self, *args, **kwargs) -> None:
        self.calls.append("warning")

    def info(self, *args, **kwargs) -> None:
        self.calls.append("info")

    def success(self, *args, **kwargs) -> None:
        self.calls.append("success")

    def error(self, *args, **kwargs) -> None:
        self.calls.append("error")

    def write(self, *args, **kwargs) -> None:
        self.calls.append("write")

    def button(self, label: str, *args, **kwargs) -> bool:
        self.calls.append(f"button:{label}")
        return self._button_responses.get(label, False)

    def checkbox(self, *args, **kwargs) -> bool:
        self.calls.append("checkbox")
        return False

    def expander(self, *args, **kwargs) -> _DummyCtx:
        self.calls.append("expander")
        return _DummyCtx()

    def progress(self, *args, **kwargs) -> _DummyCtx:
        self.calls.append("progress")
        return _DummyCtx()

    def container(self) -> _DummyCtx:
        self.calls.append("container")
        return _DummyCtx()

    def tabs(self, labels):
        self.calls.append("tabs")
        return [_DummyCtx() for _ in labels]

    def columns(self, spec):
        self.calls.append("columns")
        return [_DummyCtx() for _ in spec]

    def set_page_config(self, *args, **kwargs) -> None:
        self.calls.append("set_page_config")

    def rerun(self) -> None:
        self.calls.append("rerun")

    def spinner(self, *args, **kwargs) -> _DummyCtx:
        self.calls.append("spinner")
        return _DummyCtx()
