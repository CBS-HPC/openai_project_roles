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
    fetch_usage_by_api_key,   # <-- import your function
)

USAGE_BY_KEY_FILE = Path("openai_usage_by_api_key.csv")


# -----------------------------
# Date helpers
# -----------------------------
def _date_range_days(start_d: date, end_d: date) -> List[date]:
    days = []
    cur = start_d
    while cur <= end_d:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days


def _safe_concat_usage(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Safe concat for usage-by-key cache:
    columns: project_id, api_key_id, day, metric, value
    """
    cols = ["project_id", "api_key_id", "day", "metric", "value"]

    def _empty_typed() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "project_id": pd.Series(dtype="string"),
                "api_key_id": pd.Series(dtype="string"),
                "day": pd.Series(dtype="object"),         # python date
                "metric": pd.Series(dtype="string"),
                "value": pd.Series(dtype="float64"),
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
        out["api_key_id"] = out["api_key_id"].astype("string")
        out["metric"] = out["metric"].astype("string")
        out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.date
        out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0).astype("float64")

        out = out.dropna(subset=["day", "metric", "project_id", "api_key_id"])
        return out[cols]

    normed = [_normalize(d) for d in dfs if d is not None]
    normed = [d for d in normed if not d.empty]

    if not normed:
        return _empty_typed()
    if len(normed) == 1:
        return normed[0].copy()
    return pd.concat(normed, ignore_index=True)


# -----------------------------
# Cache IO
# -----------------------------
def _load_usage_by_key_cache_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["project_id", "api_key_id", "day", "metric", "value"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["project_id", "api_key_id", "day", "metric", "value"])

    for c in ["project_id", "api_key_id", "day", "metric", "value"]:
        if c not in df.columns:
            df[c] = np.nan

    df["project_id"] = df["project_id"].astype("string")
    df["api_key_id"] = df["api_key_id"].astype("string")
    df["metric"] = df["metric"].astype("string")
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0).astype("float64")

    df = df.dropna(subset=["day", "metric", "project_id", "api_key_id"])
    return df[["project_id", "api_key_id", "day", "metric", "value"]]


def _save_usage_by_key_cache_csv(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values(["day", "project_id", "api_key_id", "metric"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _add_coverage_rows_usage(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Coverage marker rows for each day fetched.
    """
    days = _date_range_days(start_d, end_d)
    cov = pd.DataFrame(
        {
            "project_id": pd.Series(["__coverage__"] * len(days), dtype="string"),
            "api_key_id": pd.Series(["__coverage__"] * len(days), dtype="string"),
            "day": pd.Series(days, dtype="object"),
            "metric": pd.Series(["__coverage__"] * len(days), dtype="string"),
            "value": pd.Series([0.0] * len(days), dtype="float64"),
        }
    )
    if df is None or df.empty:
        return cov.copy()
    return _safe_concat_usage(df, cov)


def _missing_day_ranges_usage(cached_df: pd.DataFrame, start_d: date, end_d: date) -> List[Tuple[date, date]]:
    cov = cached_df[
        (cached_df["project_id"] == "__coverage__")
        & (cached_df["api_key_id"] == "__coverage__")
        & (cached_df["metric"] == "__coverage__")
    ].copy()

    present_days = set(cov["day"].tolist())
    wanted = _date_range_days(start_d, end_d)
    missing = [d for d in wanted if d not in present_days]

    if not missing:
        return []

    missing = sorted(missing)
    ranges: List[Tuple[date, date]] = []
    r0 = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if d == prev + timedelta(days=1):
            prev = d
        else:
            ranges.append((r0, prev))
            r0 = d
            prev = d
    ranges.append((r0, prev))
    return ranges


# -----------------------------
# Transform API response → tidy df
# -----------------------------
def _usage_totals_dict_to_tidy_df(
    project_id: str,
    day: date,
    totals_by_key: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    totals_by_key = { api_key_id: {metric: value, ...}, ... }
    Returns tidy rows: project_id, api_key_id, day, metric, value
    """
    rows: List[Dict[str, Any]] = []
    for kid, metrics in (totals_by_key or {}).items():
        kid = str(kid or "unknown")
        for metric, val in (metrics or {}).items():
            try:
                fval = float(val)
            except Exception:
                continue
            rows.append(
                {
                    "project_id": str(project_id),
                    "api_key_id": kid,
                    "day": day,
                    "metric": str(metric),
                    "value": fval,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["project_id", "api_key_id", "day", "metric", "value"])

    df = pd.DataFrame(rows)
    return _safe_concat_usage(df)


# -----------------------------
# Pull incremental usage-by-key data
# -----------------------------
def _pull_and_store_usage_by_key_data(
    admin_api_key: str,
    months: int,
    include_archived: bool,
    project_ids: Optional[List[str]],
    usage_by_key_file_path: Path,
) -> None:
    end_dt = utc_now()
    start_dt = subtract_months(end_dt, months)
    start_d, end_d = start_dt.date(), end_dt.date()

    # Fetch projects list each time (names/status can change)
    with st.spinner("Fetching projects..."):
        projects = list_org_projects(admin_api_key, include_archived=include_archived, project_id=None)

    proj_df = pd.DataFrame(
        [
            {"project_id": str(p.get("id")), "project_name": p.get("name"), "status": p.get("status")}
            for p in projects
        ]
    )
    if proj_df.empty:
        st.session_state["usage_by_key_projects_df"] = proj_df
        return

    # Apply project filter (if any)
    if project_ids:
        proj_df = proj_df[proj_df["project_id"].isin([str(x) for x in project_ids])].copy()

    cached = _load_usage_by_key_cache_csv(usage_by_key_file_path)
    missing_ranges = _missing_day_ranges_usage(cached, start_d, end_d)

    pulled_parts: List[pd.DataFrame] = []

    if missing_ranges:
        with st.spinner(f"Pulling missing usage-by-key days ({len(missing_ranges)} range(s))..."):
            for r_start, r_end in missing_ranges:
                # For each day range, we call the API per project (keeps response manageable).
                # If you have many projects, you can optimize later by pulling all projects
                # and filtering client-side, but this is simplest + predictable.
                for pid in proj_df["project_id"].tolist():
                    r_start_dt = datetime(r_start.year, r_start.month, r_start.day, tzinfo=timezone.utc)
                    r_end_dt = datetime(r_end.year, r_end.month, r_end.day, tzinfo=timezone.utc) + timedelta(days=1)

                    # Your helper returns totals across buckets; with bucket_width=1d it sums days
                    # BUT we want per-day. To keep it simple, pull day-by-day within the range.
                    # (This avoids ambiguities in how totals are returned.)
                    cur = r_start
                    while cur <= r_end:
                        day_start = datetime(cur.year, cur.month, cur.day, tzinfo=timezone.utc)
                        day_end = day_start + timedelta(days=1)

                        totals_by_key = fetch_usage_by_api_key(
                            admin_api_key=admin_api_key,
                            start_dt=day_start,
                            end_dt=day_end,
                            project_id=pid,
                        )

                        part = _usage_totals_dict_to_tidy_df(pid, cur, totals_by_key)
                        pulled_parts.append(part)

                        cur = cur + timedelta(days=1)

                # Add coverage rows once per fetched range (independent of project)
                pulled_parts.append(_add_coverage_rows_usage(pd.DataFrame(), r_start, r_end))

    if pulled_parts:
        pulled_all = _safe_concat_usage(*pulled_parts)
        merged = _safe_concat_usage(cached, pulled_all)

        # Deduplicate by (project_id, api_key_id, day, metric): keep last
        merged = (
            merged.sort_values(["project_id", "api_key_id", "day", "metric", "value"])
            .drop_duplicates(subset=["project_id", "api_key_id", "day", "metric"], keep="last")
            .reset_index(drop=True)
        )

        _save_usage_by_key_cache_csv(usage_by_key_file_path, merged)
        cached = merged

    # Store session state for this tab
    st.session_state["usage_by_key_projects_df"] = proj_df.reset_index(drop=True)
    st.session_state["usage_by_key_cache_path"] = str(usage_by_key_file_path.resolve())
    st.session_state["usage_by_key_period"] = {"start": start_d.isoformat(), "end": end_d.isoformat()}
    st.session_state["usage_by_key_df"] = cached.copy()


# -----------------------------
# Render tab UI
# -----------------------------
def render_key_tab(
    usage_by_key_file_path: Path = USAGE_BY_KEY_FILE,
) -> None:
    st.title("🔑 Usage by API key")
    st.markdown("Explore **non-cost usage metrics** grouped by **API key** within each project.")

    admin_api_key = st.session_state.get("api_key", "")
    if not admin_api_key:
        st.warning("Enter an OpenAI Admin API key in the sidebar to view usage.")
        return

    # Controls
    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        months = st.slider("Months back", min_value=1, max_value=48, value=3, key="ubk_months")
    with colB:
        include_archived = st.checkbox("Include archived projects", value=False, key="ubk_archived")
    with colC:
        pull_clicked = st.button("📥 Pull Records", type="primary", key="ubk_pull")

    # Project selector (after we have projects)
    project_filter: Optional[List[str]] = None

    # If no prior data and not pulled
    if not pull_clicked and st.session_state.get("usage_by_key_df") is None:
        st.info("Click **Pull Records** to fetch usage grouped by API key.")
        if usage_by_key_file_path.exists():
            st.caption(f"Cache file: `{str(usage_by_key_file_path.resolve())}`")
        return

    if pull_clicked:
        try:
            # First pull without filtering, then let user filter from the loaded projects
            _pull_and_store_usage_by_key_data(
                admin_api_key=admin_api_key,
                months=months,
                include_archived=include_archived,
                project_ids=None,
                usage_by_key_file_path=usage_by_key_file_path,
            )
        except Exception as e:
            st.error(f"Failed to pull usage-by-key: {e}")
            return

    # Load from session state
    df_all: pd.DataFrame = st.session_state.get("usage_by_key_df")
    projects_df: pd.DataFrame = st.session_state.get("usage_by_key_projects_df")
    period = st.session_state.get("usage_by_key_period", {})

    if df_all is None or projects_df is None or projects_df.empty:
        st.warning("No data available yet. Click **Pull Records**.")
        return

    if period:
        st.caption(f"Period: {period.get('start')} → {period.get('end')} (UTC)")
    cache_path = st.session_state.get("usage_by_key_cache_path")
    if cache_path:
        st.caption(f"Cache file: `{cache_path}`")

    # Exclude coverage rows for display/analysis
    df = df_all[
        ~(
            (df_all["project_id"] == "__coverage__")
            & (df_all["api_key_id"] == "__coverage__")
            & (df_all["metric"] == "__coverage__")
        )
    ].copy()

    # Project multi-select
    proj_options = [
        f"{r['project_id']} — {r.get('project_name') or ''}".strip()
        for _, r in projects_df.iterrows()
    ]
    proj_id_by_label = {lbl.split(" — ")[0]: lbl.split(" — ")[0] for lbl in proj_options}  # id is first token

    selected_proj_ids = st.multiselect(
        "Projects",
        options=projects_df["project_id"].tolist(),
        default=projects_df["project_id"].tolist()[:1],
        help="Select one or more projects to inspect.",
        key="ubk_selected_projects",
    )

    df = df[df["project_id"].isin([str(x) for x in selected_proj_ids])].copy()
    if df.empty:
        st.info("No usage rows for selected projects in the cached period.")
        return

    # Metric selector
    metric_options = sorted(df["metric"].dropna().unique().tolist())
    metric = st.selectbox(
        "Metric",
        options=metric_options,
        index=0,
        help="Choose which usage metric to analyze.",
        key="ubk_metric",
    )

    # Totals table: api_key_id → total(metric)
    totals = (
        df[df["metric"] == metric]
        .groupby(["project_id", "api_key_id"], as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown("### Totals by API key")
    st.dataframe(totals, use_container_width=True, hide_index=True)

    # Chart controls
    st.markdown("### Trend")
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        bar_width_label = st.selectbox("Bar width", ["Day", "Week", "Month"], index=1, key="ubk_bar_width")
    with c2:
        start_date = st.date_input("Start date", value=pd.to_datetime(period.get("start")).date(), key="ubk_start")
    with c3:
        end_date = st.date_input("End date", value=pd.to_datetime(period.get("end")).date(), key="ubk_end")

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    # Prepare dataframe
    dff = df[df["metric"] == metric].copy()
    dff["day"] = pd.to_datetime(dff["day"], errors="coerce")
    dff = dff.dropna(subset=["day"])
    dff = dff[(dff["day"] >= pd.Timestamp(start_date)) & (dff["day"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    if dff.empty:
        st.info("No data for the selected metric and date range.")
        return

    freq_map = {"Day": "D", "Week": "W-MON", "Month": "M"}
    freq = freq_map[bar_width_label]

    periods = dff["day"].dt.to_period(freq)
    if bar_width_label == "Month":
        dff["bucket"] = periods.dt.to_timestamp(how="start")
    else:
        dff["bucket"] = periods.dt.start_time

    # Aggregate into buckets
    agg = dff.groupby(["bucket", "api_key_id"], as_index=False)["value"].sum()
    wide = agg.pivot_table(index="bucket", columns="api_key_id", values="value", fill_value=0.0).sort_index()

    fig, ax = plt.subplots()
    wide.plot(kind="bar", ax=ax, stacked=True)

    # x labels
    if bar_width_label == "Day":
        labels = [idx.strftime("%Y-%m-%d") for idx in wide.index]
    elif bar_width_label == "Week":
        labels = [f"Week {idx.isocalendar().week}" for idx in wide.index]
    else:
        labels = [idx.strftime("%B") for idx in wide.index]

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by API key (stacked)")
    ax.legend(title="api_key_id", fontsize="small")
    fig.tight_layout()
    st.pyplot(fig)
