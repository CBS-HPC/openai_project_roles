from __future__ import annotations
from pathlib import Path
import streamlit as st
import requests

from common import (
    ROLES_CONFIG_PATH,
    GITHUB_RAW_URL,
    load_roles_config,
    download_roles_config,
    browse_for_yaml,
    create_project_role,
)


def render_roles_tab(roles_config_path: str = ROLES_CONFIG_PATH) -> None:
    st.title("🔑 OpenAI Project Roles Manager")
    st.markdown("Apply default project roles to OpenAI API platform projects")

    api_key = st.session_state.get("api_key", "")
    project_id = st.session_state.get("project_id", "")

    st.divider()

    if "roles_config_path" not in st.session_state:
        st.session_state.roles_config_path = roles_config_path
    roles_config_path = st.session_state.roles_config_path

    st.subheader("📝 Select Roles to Create")

    col_g1, col_g2, col_g3 = st.columns([1, 8, 1])
    with col_g1:
        st.caption("Current YAML path:")
    with col_g2:
        st.code(str(Path(roles_config_path).resolve()), language="bash")
    with col_g3:
        if st.button("Change path"):
            chosen = browse_for_yaml(
                start_path=roles_config_path, title="Select default_project_roles.yaml file"
            )
            if chosen and chosen != roles_config_path:
                test_roles = load_roles_config(roles_config_path=chosen)
                if test_roles is not None:
                    st.session_state.roles_config_path = chosen
                    st.success(f"✅ Loaded {len(test_roles)} role(s) from selected file")
                    st.rerun()
                else:
                    st.error(
                        "❌ Selected file is not a valid roles YAML. Please choose another file."
                    )
            elif chosen is None:
                st.info(
                    "Native file picker requires wxPython; install it or edit the path manually."
                )

    roles = load_roles_config(roles_config_path=roles_config_path)

    if not roles:
        st.warning(
            f"⚠️ `default_project_roles.yaml` not found or invalid at: {Path(roles_config_path).resolve()}"
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📥 Download from GitHub"):
                with st.spinner("Downloading..."):
                    if download_roles_config(GITHUB_RAW_URL, roles_config_path):
                        st.success("✅ Configuration downloaded successfully!")
                        st.rerun()
        st.info(
            "👆 Click the button above to download the configuration file, "
            f"or place `default_project_roles.yaml` at:\n\n`{Path(roles_config_path).resolve()}`"
        )
        return

    if "selected_roles" not in st.session_state:
        st.session_state.selected_roles = {role["role_name"]: False for role in roles}

    for role in roles:
        role_name = role["role_name"]
        description = role.get("description", "No description available")
        permissions = role.get("permissions", [])

        c1, c2 = st.columns([0.5, 9.5])
        with c1:
            selected = st.checkbox(
                "✓",
                value=st.session_state.selected_roles[role_name],
                key=f"check_{role_name}",
                label_visibility="collapsed",
            )
            st.session_state.selected_roles[role_name] = selected

        with c2:
            with st.expander(f"**{role_name}** - {description}", expanded=False):
                st.markdown("**Permissions:**")
                st.code("\n".join(permissions), language="text")

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_count = sum(st.session_state.selected_roles.values())
        if st.button(
            f"🚀 Create {selected_count} Selected Role(s)",
            type="primary",
            disabled=selected_count == 0 or not project_id or not api_key,
            width="stretch",
        ):
            if not api_key:
                st.error("❌ Please provide an API key")
                return
            if not project_id:
                st.error("❌ Please provide a project ID")
                return

            selected_roles_list = [
                role for role in roles if st.session_state.selected_roles[role["role_name"]]
            ]

            progress_bar = st.progress(0)
            status_container = st.container()

            for idx, role_def in enumerate(selected_roles_list):
                role_name = role_def["role_name"]
                with status_container:
                    st.write(f"Creating role: **{role_name}**...")

                try:
                    result = create_project_role(project_id, role_def, api_key)
                    with status_container:
                        st.success(f"✅ Created: {role_name} (ID: {result.get('id', 'N/A')})")
                except requests.exceptions.HTTPError as e:
                    with status_container:
                        st.error(f"❌ Failed to create {role_name}: {str(e)}")
                except Exception as e:
                    with status_container:
                        st.error(f"❌ Error creating {role_name}: {str(e)}")

                progress_bar.progress((idx + 1) / len(selected_roles_list))
