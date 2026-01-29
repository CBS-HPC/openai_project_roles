"""CLI entry point for the OpenAI Project Roles app."""

import os
import sys
from streamlit.web import cli as stcli


def main():
    """Launch the Streamlit app with optional script args.

    Usage:
        project-roles
        project-roles --roles-config-path path/to/default_project_roles.yaml
        project-roles --budget-path path/to/openai_project_budgets.yaml
        project-roles --usage-path path/to/openai_project_usage.csv
        project-roles -c path/to/default_project_roles.yaml
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(module_dir, "app/main.py")

    # Parse CLI args for this wrapper (sys.argv[0] is the script name)
    args = sys.argv[1:]
    roles_config_path = None
    budget_path = None
    usage_path = None
    passthrough_args = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--roles-config-path", "-c") and i + 1 < len(args):
            roles_config_path = args[i + 1]
            i += 2
        elif arg == "--budget-path" and i + 1 < len(args):
            budget_path = args[i + 1]
            i += 2
        elif arg == "--usage-path" and i + 1 < len(args):
            usage_path = args[i + 1]
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

    # If paths were provided, pass them as script args after the "--"
    # so app.py can read it from sys.argv (e.g. via argparse).
    script_args = []
    if roles_config_path:
        script_args.extend(["--roles-config-path", roles_config_path])
    if budget_path:
        script_args.extend(["--budget-path", budget_path])
    if usage_path:
        script_args.extend(["--usage-path", usage_path])
    if script_args:
        streamlit_args.extend(["--", *script_args])

    # (Optional) also forward any other args to streamlit itself
    # streamlit_args.extend(passthrough_args)

    sys.argv = streamlit_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
