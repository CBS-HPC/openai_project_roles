import argparse
import sys
import streamlit as st

from common import ROLES_CONFIG_PATH, BUDGETS_FILE, USAGE_FILE, render_credentials_section
from tab_roles import render_roles_tab
from tab_usage import render_usage_tab


def main(
    roles_config_path: str = ROLES_CONFIG_PATH,
    budget_path: str = BUDGETS_FILE,
    usage_path: str = USAGE_FILE,
) -> None:
    st.set_page_config(page_title="OpenAI Project Roles Manager", page_icon="🔑", layout="wide")

    render_credentials_section()

    tab_roles, tab_usage = st.tabs(["🔑 Project Roles Manager", "📊 Project Usage"])
    # tab_roles, tab_usage, tab_key = st.tabs(["🔑 Project Roles Manager", "📊 Project Usage", "📊 API Key Usage"])

    with tab_roles:
        render_roles_tab(roles_config_path=roles_config_path)

    with tab_usage:
        render_usage_tab(
            budget_file_path=budget_path,
            usage_file_path=usage_path,
        )

    # with tab_key:
    #     render_key_tab()


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--roles-config-path",
        default=str(ROLES_CONFIG_PATH),
        help="Path to default_project_roles.yaml",
    )
    parser.add_argument(
        "--budget-path",
        default=str(BUDGETS_FILE),
        help="Path to budgets file (JSON/CSV depending on your implementation).",
    )
    parser.add_argument(
        "--usage-path",
        default=str(USAGE_FILE),
        help="Path to usage cache CSV (daily project usage/cost cache).",
    )

    # Streamlit will pass its own args; parse_known_args avoids breaking on them.
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


if __name__ == "__main__":
    args = _parse_script_args()
    main(
        roles_config_path=args.roles_config_path,
        budget_path=args.budget_path,
        usage_path=args.usage_path,
    )
