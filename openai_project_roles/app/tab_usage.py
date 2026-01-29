from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from common import (
    utc_now,
    subtract_months,
    list_org_projects,
    fetch_costs_by_project,
    load_project_budgets,
    save_project_budgets,
    BUDGETS_FILE,
    USAGE_FILE,
)


# -----------------------------
# Small utilities
# -----------------------------
def _scalarize(x: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _compute_pct_of_budget(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pct_of_budget"] = np.where(
        out["budget_usd"].notna() & (out["budget_usd"] > 0),
        (out["spend_usd"] / out["budget_usd"]) * 100.0,
        np.nan,
    ).round(2)
    return out


def _to_day_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _date_range_days(start_d: date, end_d: date) -> List[date]:
    """Inclusive date list."""
    days = []
    cur = start_d
    while cur <= end_d:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days


# -----------------------------
# Usage CSV cache helpers
# -----------------------------
def _load_usage_cache_csv(usage_file_path: Path) -> pd.DataFrame:
    """
    CSV schema: project_id, day, cost_usd
    day stored as YYYY-MM-DD.
    """
    if not usage_file_path.exists():
        return pd.DataFrame(columns=["project_id", "day", "cost_usd"])

    try:
        df = pd.read_csv(usage_file_path)
    except Exception:
        return pd.DataFrame(columns=["project_id", "day", "cost_usd"])

    # Normalize
    for col in ["project_id", "day", "cost_usd"]:
        if col not in df.columns:
            df[col] = np.nan

    df["project_id"] = df["project_id"].astype(str)
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["day"])

    # Drop the most recent day to force a partial-day refresh on next pull.
    if not df.empty:
        last_day = df["day"].max()
        df = df[df["day"] != last_day]

    return df[["project_id", "day", "cost_usd"]]


def _save_usage_cache_csv(usage_file_path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values(["day", "project_id"]).reset_index(drop=True)
    usage_file_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(usage_file_path, index=False)


def _daily_costs_dict_to_df(daily_costs: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pid, day_map in (daily_costs or {}).items():
        for day, val in (day_map or {}).items():
            rows.append({"project_id": str(pid), "day": day, "cost_usd": float(val)})
    if not rows:
        return pd.DataFrame(columns=["project_id", "day", "cost_usd"])

    df = pd.DataFrame(rows)
    df["project_id"] = df["project_id"].astype(str)
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["day"])
    return df[["project_id", "day", "cost_usd"]]


def _df_to_daily_costs_dict(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Convert df(project_id, day=date, cost_usd) to dict {pid: {YYYY-MM-DD: cost}}.
    Keeps only non-zero values (matches your earlier requirement).
    """
    out: Dict[str, Dict[str, float]] = {}
    if df is None or df.empty:
        return out

    tmp = df.copy()
    tmp["project_id"] = tmp["project_id"].astype(str)
    tmp["day_str"] = pd.to_datetime(tmp["day"], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp["cost_usd"] = pd.to_numeric(tmp["cost_usd"], errors="coerce").fillna(0.0)

    tmp = tmp[tmp["cost_usd"] != 0.0]

    for pid, g in tmp.groupby("project_id"):
        out[pid] = {r["day_str"]: float(r["cost_usd"]) for _, r in g.iterrows()}
    return out


def _missing_day_ranges(
    cached_df: pd.DataFrame,
    start_d: date,
    end_d: date,
    project_id_filter: Optional[str],
) -> List[Tuple[date, date]]:
    """
    Determine missing contiguous day ranges (inclusive) within [start_d, end_d].

    We treat a day as "present" if ANY row exists for that day in the cache
    (this is sufficient because the API returns per-project rows only for non-zero;
     days with all-zero would have no rows, so we must explicitly mark days as complete
     when we fetch them).
    """
    # We need a "coverage" table that remembers which days were fetched.
    # Since the daily costs data omits zeros, we store a special marker row:
    # project_id="__coverage__", cost_usd=0 for each fetched day.
    cov = cached_df[cached_df["project_id"] == "__coverage__"].copy()
    present_days = set(cov["day"].tolist())

    wanted_days = _date_range_days(start_d, end_d)
    missing_days = [d for d in wanted_days if d not in present_days]

    if not missing_days:
        return []

    # Collapse to contiguous ranges
    missing_days = sorted(missing_days)
    ranges: List[Tuple[date, date]] = []
    r_start = missing_days[0]
    prev = missing_days[0]
    for d in missing_days[1:]:
        if d == prev + timedelta(days=1):
            prev = d
        else:
            ranges.append((r_start, prev))
            r_start = d
            prev = d
    ranges.append((r_start, prev))
    return ranges


def _add_coverage_rows(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """Add coverage marker rows for each day in [start_d, end_d]."""
    days = _date_range_days(start_d, end_d)

    cov = pd.DataFrame(
        {
            "project_id": pd.Series(["__coverage__"] * len(days), dtype="string"),
            "day": pd.Series(days, dtype="object"),  # dates
            "cost_usd": pd.Series([0.0] * len(days), dtype="float64"),
        }
    )

    # If df is empty, just return coverage rows (avoids concat dtype warning)
    if df is None or df.empty:
        return cov[["project_id", "day", "cost_usd"]].copy()

    # Ensure df has the right columns and stable dtypes before concat
    out_df = df.copy()
    for col in ["project_id", "day", "cost_usd"]:
        if col not in out_df.columns:
            out_df[col] = np.nan

    out_df["project_id"] = out_df["project_id"].astype("string")
    out_df["day"] = pd.to_datetime(out_df["day"], errors="coerce").dt.date
    out_df["cost_usd"] = pd.to_numeric(out_df["cost_usd"], errors="coerce").fillna(0.0).astype("float64")

    # Drop rows that are totally NA in the key fields (optional but keeps concat clean)
    out_df = out_df.dropna(subset=["day"])

    return pd.concat([out_df[["project_id", "day", "cost_usd"]], cov], ignore_index=True)


# -----------------------------
# Session state helpers
# -----------------------------
def _ensure_usage_defaults(budget_file_path: str, usage_file_path: Path) -> None:
    if "budget_file_path" not in st.session_state:
        st.session_state.budget_file_path = budget_file_path
    if "usage_file_path" not in st.session_state:
        st.session_state.usage_file_path = str(usage_file_path)

    st.session_state.setdefault("usage_plot_projects", [])
    st.session_state.setdefault("usage_params", None)


def _have_usage_data() -> bool:
    return (st.session_state.get("usage_df") is not None) and (st.session_state.get("usage_daily_costs_df") is not None)


def _maybe_invalidate_on_param_change(current_params: Dict[str, Any]) -> bool:
    last_params = st.session_state.get("usage_params")
    changed = (last_params != current_params)

    have_df = st.session_state.get("usage_df") is not None
    have_daily = st.session_state.get("usage_daily_costs_df") is not None

    if changed and (have_df or have_daily):
        # We do NOT delete the CSV cache here. We only clear session-state views.
        st.session_state["usage_df"] = None
        st.session_state["usage_daily_costs"] = None
        st.session_state["usage_daily_costs_df"] = None
        st.session_state["usage_daily_costs_meta"] = None
        st.session_state["usage_plot_projects"] = []
        return True

    return False


# -----------------------------
# UI controls
# -----------------------------
def _render_usage_controls() -> Tuple[int, bool, bool, Dict[str, Any]]:
    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        months = st.slider("Months back", min_value=1, max_value=48, value=6)
    with colB:
        include_archived = st.checkbox("Include archived projects", value=False)
    with colC:
        pull_clicked = st.button("📥 Pull Records", type="primary")

    current_params = {"months": months, "include_archived": include_archived}
    return months, include_archived, pull_clicked, current_params



# Add this helper somewhere near your other helpers (top of file)

def _safe_concat_daily(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Safely concatenate daily usage frames with stable dtypes.
    Avoids FutureWarning when inputs are empty or all-NA.
    Expected columns: project_id, day, cost_usd
    """
    cols = ["project_id", "day", "cost_usd"]

    def _empty_typed() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "project_id": pd.Series(dtype="string"),
                "day": pd.Series(dtype="object"),      # python date objects
                "cost_usd": pd.Series(dtype="float64"),
            }
        )

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return _empty_typed()

        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan

        out["project_id"] = out["project_id"].astype("string")
        out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.date
        out["cost_usd"] = (
            pd.to_numeric(out["cost_usd"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )

        out = out.dropna(subset=["day"])
        return out[cols]

    normed = [_normalize(d) for d in dfs if d is not None]
    normed = [d for d in normed if not d.empty]

    if not normed:
        return _empty_typed()

    if len(normed) == 1:
        return normed[0].copy()

    return pd.concat(normed, ignore_index=True)


# -----------------------------
# Data pull/build (incremental)
# -----------------------------
def _pull_and_store_usage_data(
    api_key: str,
    months: int,
    include_archived: bool,
    project_id: Optional[str],
    budget_file_path: str,
    usage_file_path: Path,
    current_params: Dict[str, Any],
) -> None:
    """
    Incremental pull:
      - Determine requested [start_dt, end_dt] based on months
      - Load usage cache CSV
      - Determine missing day ranges using coverage markers
      - Fetch only missing ranges
      - Merge & save cache
      - Build session_state tables from cached data (no full re-pull needed)
    """
    list_org_projects.clear()
    fetch_costs_by_project.clear()

    st.session_state["usage_params"] = current_params

    end_dt = utc_now()
    start_dt = subtract_months(end_dt, months)

    # work with whole days (UTC)
    start_d = start_dt.date()
    end_d = end_dt.date()

    budgets = load_project_budgets(budget_file=budget_file_path)

    # Projects list can change, so we still fetch projects each pull
    with st.spinner("Fetching projects..."):
        projects = list_org_projects(
            api_key,
            include_archived=include_archived,
            project_id=project_id if project_id else None,
        )
    if project_id and not projects:
        raise ValueError(f"Project `{project_id}` not found.")

    cached = _load_usage_cache_csv(usage_file_path)

    # Determine missing day ranges (via coverage markers)
    missing_ranges = _missing_day_ranges(
        cached_df=cached,
        start_d=start_d,
        end_d=end_d,
        project_id_filter=project_id,
    )
  
    # Fetch only missing day ranges
    pulled_daily_parts: List[pd.DataFrame] = []
    if missing_ranges:
        with st.spinner(f"Pulling missing usage days ({len(missing_ranges)} range(s))..."):
            for r_start, r_end in missing_ranges:
                # Convert to datetimes (inclusive day range)
                r_start_dt = datetime(r_start.year, r_start.month, r_start.day, tzinfo=timezone.utc)
                r_end_dt = datetime(r_end.year, r_end.month, r_end.day, tzinfo=timezone.utc) + timedelta(days=1)

                # fetch for [r_start_dt, r_end_dt+1day) - aligns with API end_time semantics
                _, daily_costs = fetch_costs_by_project(
                    api_key,
                    start_dt=r_start_dt,
                    end_dt=r_end_dt,
                    project_id=project_id if project_id else None,
                )

                part_df = _daily_costs_dict_to_df(daily_costs)
                # add coverage markers for the fetched interval
                part_df = _add_coverage_rows(part_df, r_start, r_end)
                pulled_daily_parts.append(part_df)

    # Merge cache with pulled
    if pulled_daily_parts:
        #pulled_all = pd.concat(pulled_daily_parts, ignore_index=True)
        #merged = pd.concat([cached, pulled_all], ignore_index=True)
        pulled_all = _safe_concat_daily(*pulled_daily_parts)
        merged = _safe_concat_daily(cached, pulled_all)

        # dedupe: keep max cost_usd for same (project_id, day) (safe)
        merged["project_id"] = merged["project_id"].astype(str)
        merged["day"] = pd.to_datetime(merged["day"], errors="coerce").dt.date
        merged["cost_usd"] = pd.to_numeric(merged["cost_usd"], errors="coerce").fillna(0.0)
        merged = merged.dropna(subset=["day"])
        merged = merged.sort_values(["project_id", "day", "cost_usd"]).drop_duplicates(
            subset=["project_id", "day"],
            keep="last",
        )
        _save_usage_cache_csv(usage_file_path, merged)
        cached = merged

    # Build "current period" daily df from cache (exclude coverage rows)
    period_daily = cached[(cached["day"] >= start_d) & (cached["day"] <= end_d)].copy()
    period_daily = period_daily[period_daily["project_id"] != "__coverage__"]

    # Session state daily df
    period_daily_df = period_daily.copy()
    period_daily_df["day"] = pd.to_datetime(period_daily_df["day"], errors="coerce")
    period_daily_df["cost_usd"] = pd.to_numeric(period_daily_df["cost_usd"], errors="coerce").fillna(0.0)
    st.session_state["usage_daily_costs_df"] = period_daily_df[["project_id", "day", "cost_usd"]]

    # Session state daily dict (non-zero only)
    st.session_state["usage_daily_costs"] = _df_to_daily_costs_dict(period_daily)

    st.session_state["usage_daily_costs_meta"] = {
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "params": current_params,
        "project_id_filter": (project_id if project_id else None),
        "cache_file": str(usage_file_path.resolve()),
        "missing_ranges_pulled": [(a.isoformat(), b.isoformat()) for a, b in missing_ranges],
    }

    # ---- Build budgets/spend overview table from period_daily_df totals ----
    spend_by_project = (
        period_daily.groupby("project_id", as_index=False)["cost_usd"].sum()
        if not period_daily.empty
        else pd.DataFrame(columns=["project_id", "cost_usd"])
    )
    spend_map = {str(r["project_id"]): float(r["cost_usd"]) for _, r in spend_by_project.iterrows()}

    rows: List[Dict[str, Any]] = []
    for p in projects:
        pid = str(p.get("id"))
        spend = float(spend_map.get(pid, 0.0))
        budget = budgets.get(pid, None)

        pct = None
        if budget is not None and budget > 0:
            pct = (spend / budget) * 100.0

        created_at = p.get("created_at")
        created_dt = ""
        if isinstance(created_at, (int, float)):
            created_dt = datetime.fromtimestamp(created_at, tz=timezone.utc).strftime("%Y-%m-%d")

        rows.append(
            {
                "project_id": pid,
                "project_name": p.get("name"),
                "status": p.get("status"),
                "created_at": created_dt,
                "spend_usd": round(spend, 2),
                "budget_usd": budgets.get(pid, None),
                "pct_of_budget": (round(pct, 2) if pct is not None else None),
            }
        )

    df = pd.DataFrame(rows)

    # Add unattributed (if present in daily)
    unattributed_spend = float(spend_map.get("unattributed", 0.0))
    if unattributed_spend != 0.0 and "unattributed" not in set(df["project_id"].astype(str)):
        row = pd.DataFrame([{
            "project_id": "unattributed",
            "project_name": "(unattributed)",
            "status": "",
            "created_at": "",
            "spend_usd": round(unattributed_spend, 4),
            "budget_usd": budgets.get("unattributed"),
            "pct_of_budget": None,
        }])
        df = pd.concat([df, row], ignore_index=True)

    df = df.sort_values("spend_usd", ascending=False).reset_index(drop=True)

    st.session_state["usage_df"] = df
    st.session_state["usage_period"] = {"start": start_d.isoformat(), "end": end_d.isoformat()}

    # reset chart selection after a fresh pull
    st.session_state["usage_plot_projects"] = []


# -----------------------------
# Chart selection
# -----------------------------
def _render_chart_selector(df: pd.DataFrame) -> List[str]:
    st.markdown("### Usage chart ")

    options = []
    id_by_label = {}
    for _, r in df.iterrows():
        pid = str(r["project_id"])
        name = str(r.get("project_name") or pid)
        label = f"{pid} — {name} "
        options.append(label)
        id_by_label[label] = pid

    selected_ids = set(map(str, st.session_state.get("usage_plot_projects", [])))
    default_labels = [lbl for lbl, pid in id_by_label.items() if pid in selected_ids]

    selected_labels: List[str] = st.multiselect(
        "Show usage chart for projects",
        options=options,
        default=default_labels,
        help="Select one or more projects to visualize.",
        key="usage_chart_project_multiselect",
    )

    selected_pids = [id_by_label[lbl] for lbl in selected_labels]
    st.session_state["usage_plot_projects"] = selected_pids
    return selected_pids


# -----------------------------
# Budgets table (editing only)
# -----------------------------
def _render_budgets_table(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### Budgets Table")

    df = df.copy()

    df["budget_usd"] = df["budget_usd"].apply(_scalarize)
    df["budget_usd"] = pd.to_numeric(df["budget_usd"], errors="coerce")
    df["spend_usd"] = pd.to_numeric(df["spend_usd"], errors="coerce")

    df = _compute_pct_of_budget(df)

    edited = st.data_editor(
        df,
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        column_config={
            "spend_usd": st.column_config.NumberColumn("Spend (USD)", format="%.4f"),
            "budget_usd": st.column_config.NumberColumn("Budget (USD)", min_value=0, step=1, format="%d"),
            "pct_of_budget": st.column_config.NumberColumn("% of Budget", format="%.2f"),
        },
        disabled=["project_id", "project_name", "status", "created_at", "spend_usd", "pct_of_budget"],
        key="usage_budgets_editor",
    )

    edited["budget_usd"] = edited["budget_usd"].apply(_scalarize)
    edited["budget_usd"] = pd.to_numeric(edited["budget_usd"], errors="coerce")
    edited = _compute_pct_of_budget(edited)

    return edited


def _budgets_from_edited_table(edited: pd.DataFrame) -> Dict[str, float]:
    new_budgets: Dict[str, float] = {}
    for _, row in edited.iterrows():
        pid = row["project_id"]
        b = row.get("budget_usd", None)
        if b is None or (isinstance(b, float) and (b != b)):  # NaN
            continue
        try:
            new_budgets[str(pid)] = float(b)
        except Exception:
            continue
    return new_budgets


def _render_save_budgets_controls(budget_file_path: str, new_budgets: Dict[str, float]) -> None:
    col_save1, col_save2 = st.columns([1, 6])
    with col_save1:
        save_clicked = st.button("💾 Save budgets", type="secondary")
    with col_save2:
        st.caption(f"Budgets file: `{str(Path(budget_file_path).resolve())}`")

    if save_clicked:
        try:
            save_project_budgets(budgets=new_budgets, budget_file=budget_file_path)
            st.success("✅ Budgets saved to local file.")
        except Exception as e:
            st.error(f"❌ Failed to save budgets: {e}")


def _render_csv_download(edited: pd.DataFrame, months: int) -> None:
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download table as CSV",
        data=csv_bytes,
        file_name=f"openai_project_usage_{months}m.csv",
        mime="text/csv",
    )


def _render_usage_cache_info(usage_file_path: Path) -> None:
    # Helpful UI hint (optional)
    if usage_file_path.exists():
        st.caption(f"Usage cache file: `{str(usage_file_path.resolve())}`")


# -----------------------------
# Chart
# -----------------------------
def _render_usage_chart(selected_pids: List[str]) -> None:
    if not selected_pids:
        return

    daily_df = st.session_state.get("usage_daily_costs_df")
    meta = st.session_state.get("usage_daily_costs_meta") or {}
    period2 = st.session_state.get("usage_period") or {}

    if daily_df is None or daily_df.empty:
        st.info("Daily usage data is not available yet. Click **Pull Records** again.")
        return

    default_start = pd.to_datetime(period2.get("start"), errors="coerce")
    default_end = pd.to_datetime(period2.get("end"), errors="coerce")
    if pd.isna(default_start):
        default_start = pd.to_datetime(meta.get("start"), errors="coerce")
    if pd.isna(default_end):
        default_end = pd.to_datetime(meta.get("end"), errors="coerce")

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        bar_width_label: str = st.selectbox(
            "Bar width",
            ["Day", "Week", "Month"],
            index=1,
            key="usage_chart_bar_width",
        )
    with c2:
        start_date = st.date_input(
            "Start date",
            value=(default_start.date() if default_start is not None and not pd.isna(default_start) else utc_now().date()),
            key="usage_chart_start_date",
        )
    with c3:
        end_date = st.date_input(
            "End date",
            value=(default_end.date() if default_end is not None and not pd.isna(default_end) else utc_now().date()),
            key="usage_chart_end_date",
        )

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    freq_map = {"Day": "D", "Week": "W-MON", "Month": "M"}
    freq = freq_map[bar_width_label]

    dff = daily_df.copy()
    dff["project_id"] = dff["project_id"].astype(str)
    dff = dff[dff["project_id"].isin([str(x) for x in selected_pids])]
    dff = dff.dropna(subset=["day", "cost_usd"])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    dff = dff[(dff["day"] >= start_ts) & (dff["day"] <= end_ts)]

    if dff.empty:
        st.info("No usage data for the selected projects in the chosen date range.")
        return

    periods = dff["day"].dt.to_period(freq)
    if bar_width_label == "Month":
        dff["bucket"] = periods.dt.to_timestamp(how="start")
    else:
        dff["bucket"] = periods.dt.start_time

    agg = dff.groupby(["bucket", "project_id"], as_index=False)["cost_usd"].sum()
    wide = agg.pivot_table(index="bucket", columns="project_id", values="cost_usd", fill_value=0.0).sort_index()

    fig, ax = plt.subplots()
    wide.plot(kind="bar", ax=ax, stacked=True)

    if bar_width_label == "Day":
        labels = [idx.strftime("%Y-%m-%d") for idx in wide.index]
    elif bar_width_label == "Week":
        labels = [f"Week {idx.isocalendar().week}" for idx in wide.index]
    else:
        labels = [idx.strftime("%B") for idx in wide.index]

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Usage")
    ax.legend(title="Project", fontsize="small")
    fig.tight_layout()
    st.pyplot(fig)


# -----------------------------
# Main tab function
# -----------------------------
def render_usage_tab(
    budget_file_path: str = BUDGETS_FILE,
    usage_file_path: Path = USAGE_FILE,
) -> None:
    st.title("📊 Project Usage Overview")
    st.markdown("View org project spend (USD) and manage per-project budgets (persisted locally).")

    api_key = st.session_state.get("api_key", "")
    project_id = st.session_state.get("project_id", "")

    if not api_key:
        st.warning("Enter an OpenAI Admin API key in the sidebar to view usage.")
        return

    _ensure_usage_defaults(budget_file_path, usage_file_path)
    budget_file_path = st.session_state.budget_file_path
    usage_file_path = Path(st.session_state.get("usage_file_path", str(usage_file_path)))

    months, include_archived, pull_clicked, current_params = _render_usage_controls()
    _maybe_invalidate_on_param_change(current_params)

    have_data = _have_usage_data()

    if not pull_clicked and not have_data:
        _render_usage_cache_info(usage_file_path)
        st.info("Click **Pull Records** to fetch the project list and usage costs for the selected period.")
        return

    if pull_clicked:
        try:
            _pull_and_store_usage_data(
                api_key=api_key,
                months=months,
                include_archived=include_archived,
                project_id=(project_id if project_id else None),
                budget_file_path=budget_file_path,
                usage_file_path=usage_file_path,
                current_params=current_params,
            )
        except Exception as e:
            st.error(f"Failed to pull usage: {e}")
            return

    df = st.session_state.get("usage_df")
    if df is None or df.empty:
        st.warning("No usage data available yet. Click **Pull Records**.")
        return

    period = st.session_state.get("usage_period", {})
    if period:
        st.caption(f"Period: {period.get('start')} → {period.get('end')} (UTC)")

    _render_usage_cache_info(usage_file_path)

    edited = _render_budgets_table(df)
    st.session_state["usage_df"] = edited

    new_budgets = _budgets_from_edited_table(edited)
    _render_save_budgets_controls(budget_file_path, new_budgets)

    _render_csv_download(edited, months)

    selected_pids = _render_chart_selector(df)
    _render_usage_chart(selected_pids)
