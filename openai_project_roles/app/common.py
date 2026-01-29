from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone, timedelta


import requests
import streamlit as st
import yaml

# ----------------------------
# Config
# ----------------------------
BASE_URL = "https://api.openai.com/v1"
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/default_project_roles.yaml"
)

# Local persistence (plain text; use only on trusted machines)
PACKAGE_DIR = Path(__file__).resolve().parents[1]
ADMIN_KEY_FILE = Path(".env")
ROLES_CONFIG_PATH = str(PACKAGE_DIR / "default_project_roles.yaml")
BUDGETS_FILE = PACKAGE_DIR / "openai_project_budgets.yaml"
USAGE_FILE = PACKAGE_DIR / "openai_project_usage.csv"


# Optional: native file dialog
try:
    import wx

    _WX_AVAILABLE = True
except Exception:
    _WX_AVAILABLE = False


# ----------------------------
# Persistence: Admin Key
# ----------------------------
def load_persisted_admin_key() -> str:
    """Load OPENAI_ADMIN_KEY from env or from local file."""
    env_key = os.environ.get("OPENAI_ADMIN_KEY")
    if env_key:
        return env_key

    if ADMIN_KEY_FILE.exists():
        try:
            key = ADMIN_KEY_FILE.read_text(encoding="utf-8").strip()
            if key:
                os.environ["OPENAI_ADMIN_KEY"] = key
                return key
        except Exception:
            pass
    return ""


def persist_admin_key(key: str) -> None:
    """Persist admin key to env + local file (plain text)."""
    key = key.strip()
    if not key:
        return
    os.environ["OPENAI_ADMIN_KEY"] = key
    try:
        ADMIN_KEY_FILE.write_text(key, encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to persist admin key: {e}")


# ----------------------------
# Persistence: Budgets
# ----------------------------


def load_project_budgets(budget_file: Union[str, Path] = BUDGETS_FILE) -> Dict[str, float]:
    """Load budgets mapping from a YAML file. Expected format: {'budgets': {project_id: number}}."""
    budget_path = Path(budget_file)

    if not budget_path.exists():
        return {}

    try:
        data = yaml.safe_load(budget_path.read_text(encoding="utf-8")) or {}
        budgets = data.get("budgets") or {}
        if not isinstance(budgets, dict):
            return {}

        out: Dict[str, float] = {}
        for k, v in budgets.items():
            try:
                if v is None:
                    continue
                out[str(k)] = float(v)
            except Exception:
                # skip unparseable values
                continue

        return out

    except Exception:
        # keep it silent and safe in production; optionally show st.warning for debugging
        return {}


def save_project_budgets(
    budgets: Dict[str, float],
    budget_file: Union[str, Path] = BUDGETS_FILE,
) -> None:
    """Save budgets mapping to a YAML file in the format: {'budgets': {project_id: number}}."""
    budget_path = Path(budget_file)

    payload = {"budgets": {str(k): float(v) for k, v in budgets.items() if v is not None}}

    try:
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        budget_path.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as e:
        st.error(f"Failed to save budgets file: {e}")


# ----------------------------
# Date helpers
# ----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_unix_seconds(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def subtract_months(dt: datetime, months: int) -> datetime:
    """Subtract months without extra deps; clamp day if needed."""
    year = dt.year
    month = dt.month - months
    while month <= 0:
        month += 12
        year -= 1

    day = dt.day
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=dt.tzinfo)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=dt.tzinfo)
    last_day = (next_month - timedelta(days=1)).day
    day = min(day, last_day)

    return dt.replace(year=year, month=month, day=day)


# ----------------------------
# Roles config loading
# ----------------------------
def download_roles_config(github_url: str, save_path: str) -> bool:
    try:
        response = requests.get(github_url, timeout=10)
        response.raise_for_status()
        Path(save_path).write_text(response.text, encoding="utf-8")
        return True
    except Exception:
        return False


def load_roles_config(roles_config_path: str = ROLES_CONFIG_PATH) -> List[Dict[str, Any]]:
    """
    Load roles configuration with fallback strategy:
    1) local file
    2) download from GitHub
    """
    if Path(roles_config_path).exists():
        try:
            data = yaml.safe_load(Path(roles_config_path).read_text(encoding="utf-8")) or {}
            return data.get("roles", []) or []
        except Exception as e:
            st.warning(f"Local file exists but failed to load: {str(e)}")

    if download_roles_config(GITHUB_RAW_URL, roles_config_path):
        try:
            data = yaml.safe_load(Path(roles_config_path).read_text(encoding="utf-8")) or {}
            st.success("✅ Configuration downloaded from GitHub successfully!")
            return data.get("roles", []) or []
        except Exception as e:
            st.warning(f"Downloaded file but failed to load: {str(e)}")

    return []


# ----------------------------
# Native file picker (optional)
# ----------------------------
def browse_for_yaml(
    start_path: Optional[Union[str, Path]] = None, title: str = "Select a YAML file"
) -> Optional[str]:
    """Open a native file chooser dialog via wxPython if available."""
    if not _WX_AVAILABLE:
        return None

    start_str = os.fspath(start_path) if start_path else os.getcwd()
    ext = "yaml"
    wildcard = f"{ext.upper()} files (*.{ext})|*.{ext}"

    app = wx.App(False)
    if os.path.isfile(start_str):
        default_dir, default_file = os.path.split(start_str)
    else:
        default_dir, default_file = (start_str, "")

    dlg = wx.FileDialog(
        None,
        message=title,
        defaultDir=default_dir,
        defaultFile=default_file,
        wildcard=wildcard,
        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
    )

    try:
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.GetPath()
        return None
    finally:
        dlg.Destroy()
        app.Destroy()


# ----------------------------
# OpenAI API calls
# ----------------------------
def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def create_project_role(project_id: str, role_def: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/projects/{project_id}/roles"
    payload = {
        "role_name": role_def["role_name"],
        "description": role_def.get("description", ""),
        "permissions": role_def["permissions"],
    }
    r = requests.post(url, json=payload, headers=_headers(api_key), timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def list_org_projects(
    api_key: str,
    include_archived: bool = False,
    limit: int = 100,
    project_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    GET /v1/organization/projects (paginated with `after`).

    If project_id is provided, returns at most one project (if found).
    """
    url = f"{BASE_URL}/organization/projects"
    projects: List[Dict[str, Any]] = []
    after: Optional[str] = None

    project_id = project_id.strip() if project_id else None

    while True:
        params: Dict[str, Any] = {
            "limit": limit,
            "include_archived": str(include_archived).lower(),
        }
        if after:
            params["after"] = after

        r = requests.get(url, headers=_headers(api_key), params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        page_projects = payload.get("data", []) or []

        if project_id:
            # Filter client-side and early-exit if found
            for p in page_projects:
                if p.get("id") == project_id:
                    return [p]
        else:
            projects.extend(page_projects)

        if not payload.get("has_more"):
            break

        after = payload.get("last_id")
        if not after:
            break

    # If filtering by project_id and not found, return empty list
    return projects if not project_id else []


# FIX ME
@st.cache_data(ttl=300)
def fetch_usage_by_api_key(
    admin_api_key: str,
    start_dt: datetime,
    end_dt: datetime,
    project_id: Optional[str] = None,
    usage_type: str = "completions",  # <-- NEW
) -> Dict[str, Dict[str, float]]:
    """
    Fetch org usage grouped by api_key_id (and optionally filtered to a project_id).

    usage_type must be one of the Usage API sub-endpoints, e.g.:
      - "completions"
      - "embeddings"
      - "moderations"
      - "images"
      - "audio_speeches"
      - "audio_transcriptions"
      - "vector_stores"
      - "code_interpreter_sessions"
      (see OpenAI Usage API reference)

    Returns:
      { "key_...": {"requests": ..., "input_tokens": ..., ...}, ... }
    """
    url = f"{BASE_URL}/organization/usage/{usage_type}"

    params: Dict[str, Any] = {
        "start_time": to_unix_seconds(start_dt),
        "end_time": to_unix_seconds(end_dt),
        "bucket_width": "1d",
        "group_by": ["project_id", "api_key_id"],
        "limit": 31,  # daily max is 31 per docs; adjust if you change bucket_width
    }

    page: Optional[str] = None
    totals: Dict[str, Dict[str, float]] = {}

    while True:
        if page:
            params["page"] = page
        else:
            params.pop("page", None)

        r = requests.get(url, headers=_headers(admin_api_key), params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        for bucket in payload.get("data", []) or []:
            for res in bucket.get("results", []) or []:
                pid = res.get("project_id")
                kid = res.get("api_key_id") or "unknown"

                if project_id and pid != project_id:
                    continue

                metrics = totals.setdefault(kid, {})
                for k, v in res.items():
                    if k in ("project_id", "api_key_id", "object"):
                        continue
                    if isinstance(v, (int, float)):
                        metrics[k] = metrics.get(k, 0.0) + float(v)

        if not payload.get("has_more"):
            break
        page = payload.get("next_page")
        if not page:
            break

    return totals


@st.cache_data(ttl=300)
def fetch_costs_by_project(
    api_key: str,
    start_dt: datetime,
    end_dt: datetime,
    project_id: Optional[str] = None,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    GET /v1/organization/costs grouped by project_id, daily buckets.

    Returns:
      1) totals: aggregated USD cost per project
      2) daily_costs: raw daily USD costs per project (non-zero only)
         {project_id: {YYYY-MM-DD: value}}
    """
    url = f"{BASE_URL}/organization/costs"

    totals: Dict[str, float] = {}
    daily_costs: Dict[str, Dict[str, float]] = {}

    params: Dict[str, Any] = {
        "start_time": to_unix_seconds(start_dt),
        "end_time": to_unix_seconds(end_dt),
        "bucket_width": "1d",
        "group_by": ["project_id"],
        "limit": 180,
    }

    page: Optional[str] = None
    while True:
        if page:
            params["page"] = page
        else:
            params.pop("page", None)

        r = requests.get(url, headers=_headers(api_key), params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        for bucket in payload.get("data", []) or []:
            # Bucket start time → YYYY-MM-DD
            bucket_ts = bucket.get("start_time")
            if bucket_ts is None:
                continue
            day = datetime.utcfromtimestamp(bucket_ts).date().isoformat()

            for result in bucket.get("results", []) or []:
                proj = result.get("project_id") or "unattributed"

                # Filter to single project if requested
                if project_id and proj != project_id:
                    continue

                amount = result.get("amount", {}) or {}
                value = float(amount.get("value", 0.0) or 0.0)
                currency = (amount.get("currency") or "").lower()

                if currency and currency != "usd":
                    continue

                # Aggregate totals (existing behavior)
                totals[proj] = totals.get(proj, 0.0) + value

                # Store raw daily values (non-zero only)
                if value != 0.0:
                    daily_costs.setdefault(proj, {})
                    daily_costs[proj][day] = daily_costs[proj].get(day, 0.0) + value

        if not payload.get("has_more"):
            break
        page = payload.get("next_page")
        if not page:
            break

    # Ensure requested project is present in totals
    if project_id and project_id not in totals:
        totals[project_id] = 0.0

    # Remove any projects that ended up with no non-zero days
    daily_costs = {p: d for p, d in daily_costs.items() if d}

    return totals, daily_costs


# ----------------------------
# Shared UI: credentials
# ----------------------------
def render_credentials_section() -> None:
    st.sidebar.header("🔐 Credentials")

    default_api_key = load_persisted_admin_key()
    api_key = st.sidebar.text_input(
        "OpenAI Admin API Key",
        value=st.session_state.get("api_key", default_api_key),
        type="password",
        help="Defaults to OPENAI_ADMIN_KEY environment variable (persisted between sessions)",
    )
    if api_key and api_key != default_api_key:
        persist_admin_key(api_key)
        st.sidebar.success("✅ Admin API key saved for future sessions.")
    st.session_state["api_key"] = api_key

    project_id = st.sidebar.text_input(
        "Project ID",
        value=st.session_state.get("project_id", ""),
        placeholder="proj_xxxxxxxxxxxxxxxxxxxxx",
        help="Only used when creating roles in the Roles tab",
    )
    st.session_state["project_id"] = project_id

    st.sidebar.divider()
    st.sidebar.caption("Key stored in ~/.env (plain text).")
    st.sidebar.caption("Budgets stored in ~/.openai_project_budgets.yaml (plain text).")
