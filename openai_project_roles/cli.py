"""CLI entry point for the OpenAI Project Roles app."""
import os
import sys
from streamlit.web import cli as stcli


def main():
    """Launch the Streamlit app, optionally with a custom roles_config_path.

    Usage:
        project-roles
        project-roles --roles-config-path path/to/default_project_roles.yaml
        project-roles -c path/to/default_project_roles.yaml
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(module_dir, "app.py")

    # Parse CLI args for this wrapper (sys.argv[0] is the script name)
    args = sys.argv[1:]
    roles_config_path = None
    passthrough_args = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--roles-config-path", "-c") and i + 1 < len(args):
            roles_config_path = args[i + 1]
            i += 2
        else:
            # Any other args could be passed through to streamlit if you want
            passthrough_args.append(arg)
            i += 1

    # Build the arguments for Streamlit
    streamlit_args = [
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "true",
    ]

    # If a path was provided, pass it as a script argument after the "--"
    # so app.py can read it from sys.argv (e.g. via argparse).
    if roles_config_path:
        streamlit_args.extend(
            [
                "--",                  # everything after this goes to the script
                "--roles-config-path",
                roles_config_path,
            ]
        )

    # (Optional) also forward any other args to streamlit itself
    # streamlit_args.extend(passthrough_args)

    sys.argv = streamlit_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
