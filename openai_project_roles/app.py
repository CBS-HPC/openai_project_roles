"""Streamlit app for applying default project roles to OpenAI API platform projects."""
import sys
import argparse
import os
import yaml
import requests
import streamlit as st
from pathlib import Path
import wx
from typing import Optional, Union

try:
    from importlib.resources import files  # Python 3.9+
except ImportError:  # Python < 3.9
    from importlib_resources import files


BASE_URL = "https://api.openai.com/v1"
ROLES_CONFIG_PATH = "default_project_roles.yaml"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/default_project_roles.yaml"
ADMIN_KEY_FILE = Path.home() / ".openai_admin_key"  # choose any path you like


def load_persisted_admin_key() -> str:
    """Load OPENAI_ADMIN_KEY from env or from a small local file."""
    # 1) If already in env for this process, use that
    env_key = os.environ.get("OPENAI_ADMIN_KEY")
    if env_key:
        return env_key

    # 2) Otherwise try to load from file and put it into env
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
    """Persist the admin key so it's available next time the app starts."""
    key = key.strip()
    if not key:
        return

    # Set for current process
    os.environ["OPENAI_ADMIN_KEY"] = key

    # Persist to disk (⚠️ plain text – only do this on trusted machines)
    try:
        ADMIN_KEY_FILE.write_text(key, encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to persist admin key: {e}")


def _browse_for_file(
    start_path: Optional[Union[str, Path]] = None,
    title: str = "Select a YAML file",
    file_extension: str = "yaml",
) -> Optional[str]:
    """
    Open a native file chooser dialog using wxPython, restricted to a given extension.

    Args:
        start_path:     Initial folder or file path to start from.
        title:          Dialog title.
        file_extension: File extension to filter for (e.g. "yaml" or ".yaml").

    Returns:
        Selected file path as a string, or None if cancelled.
    """
    # Normalise starting path
    if start_path:
        start_str = os.fspath(start_path)
    else:
        start_str = os.getcwd()

    # Ensure we have a clean extension (no leading dot)
    ext = file_extension.lstrip(".")
    wildcard = f"{ext.upper()} files (*.{ext})|*.{ext}"

    # Create a minimal wx App just for the dialog
    app = wx.App(False)

    # File chooser: split into dir + file if a file path is given
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
            selected_path = dlg.GetPath()
        else:
            selected_path = None
    finally:
        dlg.Destroy()
        app.Destroy()

    return selected_path


def load_bundled_config():
    """Load the bundled default configuration from the package."""
    try:
        # `files()` returns a Traversable object pointing to the package
        config_path = files("openai_project_roles") / "default_project_roles.yaml"
        config_content = config_path.read_text(encoding="utf-8")

        data = yaml.safe_load(config_content)
        return data.get("roles", [])
    except Exception as e:
        st.error(f"Failed to load bundled config: {str(e)}")
        return []


def download_roles_config(github_url: str, save_path: str):
    """Download roles configuration from GitHub."""
    try:
        response = requests.get(github_url, timeout=10)
        response.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return True
    except Exception as e:
        return False


def load_roles_config(roles_config_path: str = ROLES_CONFIG_PATH):
    """
    Load roles configuration with fallback strategy:
    1. Try local file in current directory
    2. Try downloading from GitHub
    3. Fall back to bundled package version
    
    Returns:
        tuple: (roles_list, source_type) where source_type is 'local', 'downloaded', or 'bundled'
    """
    # Try 1: Local file in current directory
    if os.path.exists(roles_config_path):
        try:
            with open(roles_config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("roles", [])
        except Exception as e:
            st.warning(f"Local file exists but failed to load: {str(e)}")
    
    # Try 2: Download from GitHub
    if download_roles_config(GITHUB_RAW_URL, roles_config_path):
        try:
            with open(roles_config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            st.success("✅ Configuration downloaded from GitHub successfully!")
            return data.get("roles", [])
        except Exception as e:
            st.warning(f"Downloaded file but failed to load: {str(e)}")

    # Try 3: Use bundled version from package
    st.info("Using bundled default configuration from package...")
    return load_bundled_config()



def create_project_role(project_id: str, role_def: dict, api_key: str) -> dict:
    """Create a single project role via OpenAI API."""
    url = f"{BASE_URL}/projects/{project_id}/roles"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "role_name": role_def["role_name"],
        "description": role_def.get("description", ""),
        "permissions": role_def["permissions"],
    }
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()


def main(roles_config_path: str = ROLES_CONFIG_PATH):
    st.set_page_config(
        page_title="OpenAI Project Roles Manager",
        page_icon="🔑",
        layout="wide"
    )
    st.title("🔑 OpenAI Project Roles Manager")
    st.markdown("Apply default project roles to OpenAI API platform projects")
    
    # API Key and Project ID input fields
    st.subheader("🔐 Credentials")
    col1, col2 = st.columns(2)
    
    with col1:
        # Load default key from env or persisted file
        default_api_key = load_persisted_admin_key()

        api_key = st.text_input(
            "OpenAI Admin API Key",
            value=default_api_key,
            type="password",
            help="Defaults to OPENAI_ADMIN_KEY environment variable (persisted between sessions)",
        )

        # If user types a new key, save & persist it
        if api_key and api_key != default_api_key:
            persist_admin_key(api_key)
            st.success("✅ Admin API key saved and will be available in future sessions.")
            
    with col2:
        project_id = st.text_input(
            "Project ID",
            placeholder="proj_xxxxxxxxxxxxxxxxxxxxx",
            help="Enter the OpenAI project ID"
        )
    
    st.divider()
    
    # --- NEW: roles config path state & UI -----------------------------
    if "roles_config_path" not in st.session_state:
        st.session_state.roles_config_path = roles_config_path


    roles_config_path = st.session_state.roles_config_path
    # ------------------------------------------------------------------
    
    # --------------------------------------------
    # Display available roles with checkboxes
    # --------------------------------------------
    st.subheader("📝 Select Roles to Create")

    # Current configurable path from session state
    roles_config_path = st.session_state.get("roles_config_path", roles_config_path)

    col_g1, col_g2, col_g3 = st.columns([1, 8, 1])

    with col_g1:
        st.caption("Current YAML path:")

    with col_g2:
        st.code(str(Path(roles_config_path).resolve()), language="bash")

    with col_g3:
        if st.button("Change path"):
            chosen = _browse_for_file(
                start_path=roles_config_path,
                title="Select default_project_roles.yaml file",
                file_extension=".yaml",
            )

            if chosen and chosen != roles_config_path:
                test_roles = load_roles_config(roles_config_path=chosen)
                if test_roles is not None:
                    st.session_state.roles_config_path = chosen
                    roles_config_path = chosen  # update local variable too
                    st.success(f"✅ Loaded {len(test_roles)} role(s) from selected file")
                    st.rerun()
                else:
                    st.error("❌ Selected file is not a valid roles YAML. Please choose another file.")


    # ------------------------------------------------------------------
    # Load or download roles configuration USING the configurable path
    # ------------------------------------------------------------------
    roles = load_roles_config(roles_config_path = roles_config_path)

    if not roles:
        st.warning(
            f"⚠️ `default_project_roles.yaml` not found or invalid at: "
            f"{Path(roles_config_path).resolve()}"
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


    # Initialize session state for selected roles
    if 'selected_roles' not in st.session_state:
        st.session_state.selected_roles = {role['role_name']: False for role in roles}
    
    # Display roles in a nice format
    for role in roles:
        role_name = role['role_name']
        description = role.get('description', 'No description available')
        permissions = role.get('permissions', [])
        
        # Create two columns: checkbox on left, expander on right
        col1, col2 = st.columns([0.5, 9.5])
        
        with col1:
            # Checkbox for selection
            selected = st.checkbox(
                "✓",
                value=st.session_state.selected_roles[role_name],
                key=f"check_{role_name}",
                label_visibility="collapsed"
            )
            st.session_state.selected_roles[role_name] = selected
        
        with col2:
            # Create expandable section for each role
            with st.expander(f"**{role_name}** - {description}", expanded=False):
                # Show permissions
                st.markdown("**Permissions:**")
                st.code("\n".join(permissions), language="text")
    
    st.divider()
    
    # Create roles button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_count = sum(st.session_state.selected_roles.values())
        if st.button(
            f"🚀 Create {selected_count} Selected Role(s)",
            type="primary",
            disabled=selected_count == 0 or not project_id or not api_key,
            use_container_width=True
        ):
            # Validate inputs
            if not api_key:
                st.error("❌ Please provide an API key")
                return
            if not project_id:
                st.error("❌ Please provide a project ID")
                return
            
            # Create selected roles
            selected_roles_list = [
                role for role in roles 
                if st.session_state.selected_roles[role['role_name']]
            ]
            
            progress_bar = st.progress(0)
            status_container = st.container()
            
            created_roles = []
            failed_roles = []
            
            for idx, role_def in enumerate(selected_roles_list):
                role_name = role_def['role_name']
                
                with status_container:
                    st.write(f"Creating role: **{role_name}**...")
                
                try:
                    result = create_project_role(project_id, role_def, api_key)
                    created_roles.append(role_name)
                    
                    with status_container:
                        st.success(f"✅ Created: {role_name} (ID: {result.get('id', 'N/A')})")
                
                except requests.exceptions.HTTPError as e:
                    failed_roles.append((role_name, str(e)))
                    with status_container:
                        st.error(f"❌ Failed to create {role_name}: {str(e)}")
                
                except Exception as e:
                    failed_roles.append((role_name, str(e)))
                    with status_container:
                        st.error(f"❌ Error creating {role_name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(selected_roles_list))
            
            # Summary
            st.divider()
            if created_roles:
                st.success(f"🎉 Successfully created {len(created_roles)} role(s)")
            if failed_roles:
                st.error(f"⚠️ Failed to create {len(failed_roles)} role(s)")
    
    # Instructions
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        ### How to use this app:

        1. **Configure settings** in the credentials section:
        - Enter your OpenAI Admin API Key  
            - The key is stored for reuse and made available as `OPENAI_ADMIN_KEY` in future sessions
        - Enter your Project ID (format: `proj_xxxxxxxxxxxxxxxxxxxxx`)

        2. **Configure the roles YAML file**:
        - Check the **Current YAML path** and adjust it if needed
        - Optionally use **Change path** to browse for a different `default_project_roles.yaml`
        - If no valid file is found, you can download one from GitHub using the provided button

        3. **Select roles** to create by expanding each role and checking the box.

        4. **Review permissions** for each role in the expandable sections.

        5. **Click "Create Selected Roles"** to apply them to your project.

        ### Notes:
        - All roles are designed for research workflows.
        - No role has organization/billing/admin permissions.
        - Roles focus on inference, RAG, fine-tuning, agent building, and evaluation.
        """)


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--roles-config-path",
        default=ROLES_CONFIG_PATH,
        help="Path to default_project_roles.yaml",
    )
    # Streamlit adds its own args, so we ignore unknown ones
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = _parse_script_args()
    main(roles_config_path=args.roles_config_path)