# OpenAI Project Roles & Usage Manager

A **Streamlit-based admin dashboard** for managing **OpenAI project roles** *and* inspecting **project usage and costs** using the **OpenAI Admin API**.

This tool is designed for **administrators and platform owners** who need visibility into:
- 🔑 Project roles and permissions
- 📊 Project-level usage and cost trends
- 🧾 Incremental, cached usage data (no unnecessary re-pulls)
- 🖥️ A simple desktop-style UI with a CLI entry point

---

## ✨ Features

### 🔑 Project Roles Manager
- Load role definitions from a YAML file
- Review permissions interactively
- Create roles in bulk via the OpenAI Admin API
- Supports:
  - Local YAML paths
  - CLI-provided paths
  - Bundled defaults

### 📊 Project Usage (Cost)
- Pull **daily project cost data** from the OpenAI Admin API
- Incremental caching to CSV (only missing days are fetched)
- Interactive:
  - Project filtering
  - Time window selection
  - Daily / weekly / monthly aggregation
- Budget tracking with editable per-project budgets
- CSV export of aggregated tables

> Note: API-key-level usage depends on OpenAI Usage API availability and may vary by organization.

---

## Installation

openai-project-roles is not published on PyPI yet. Use local wheel/source installation.

### Install from local wheel files (`/dist`)
```bash
# 1) Build artifacts (if needed)
uv build

# 2) Install wheel from local dist
pip install ./dist/openai_project_roles-1.0.0-py3-none-any.whl
```

### Install directly from GitHub dist artifact
```bash
pip install https://github.com/CBS-HPC/openai_project_roles/raw/main/dist/openai_project_roles-1.0.0-py3-none-any.whl
```

### Dependencies
Installed automatically:
- streamlit
- requests
- pyyaml
- pandas
- numpy
- matplotlib

(Python >= 3.9 recommended)

---

## 🧱 Architecture Overview

- **Streamlit UI**
- **OpenAI Admin API**
- **Local CSV caches** for usage data
- Stateless reruns with `st.session_state` coordination

```
openai_project_roles/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── tab_roles.py         # Project roles UI
│   ├── tab_usage.py         # Project usage & cost UI
│   ├── common.py            # API helpers, constants
│   └── ...
├── default_project_roles.yaml
├── pyproject.toml
└── README.md
```

---

---

## Usage

### CLI
```bash
project-roles
```

With custom paths:
```bash
project-roles \
  --roles-config-path ./roles.yaml \
  --budget-path ./budgets.json \
  --usage-path ./usage_cache.csv
```

### Python
```python
from openai_project_roles.app.main import main
main()
```

With custom paths:
```python
main(
    roles_config_path="roles.yaml",
    budget_path="budgets.json",
    usage_path="usage_cache.csv",
)
```

---

## 🔐 Credentials & Security

- Requires an **OpenAI Admin API key**
- The key is entered in the UI sidebar
- Stored **locally** (plaintext) for convenience

⚠️ **Use only on trusted machines**

---

## 📊 Usage Data & Caching

- Usage data is cached locally to CSV (default: `openai_project_usage.csv`)
- The app:
  - Detects which days are missing
  - Fetches only missing ranges
  - Preserves historical data

This dramatically reduces API calls and load time.

---

## Local Config and Data Files

Default files are stored in the package data directory:
- `default_project_roles.yaml`: default role definitions (role names + permissions) used by the Project Roles tab.
- `openai_project_budgets.yaml`: editable per-project budget configuration used by the Project Usage tab.
- `openai_project_usage.csv`: cached daily project cost rows used by the Project Usage tab.
- `openai_usage_by_api_key.csv`: cached daily usage-by-key metric rows used by the API Key Usage tab.
- `openai_api_key_names.csv`: local mapping of `api_key_id -> key_name` used to label API keys in tables/charts.

---

## CI

GitHub Actions runs tests on Windows/macOS/Linux and runs Ruff on Linux.
`wxpython` is treated as optional and is skipped in CI install steps.

---

## 🤝 Contributing

Contributions welcome!
- Bug reports
- UX improvements
- Additional usage breakdowns
- Documentation improvements

Please open an issue or pull request.
