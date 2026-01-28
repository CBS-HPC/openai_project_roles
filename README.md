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

## 🚀 Installation

### Using uv (recommended for dev)
```bash
uv sync --extra test --extra lint
```


### From a local wheel
```bash
pip install openai_project_roles-<VERSION>-py3-none-any.whl
```

### Dependencies
Installed automatically:
- streamlit
- requests
- pyyaml
- pandas
- numpy
- matplotlib

(Python ≥ 3.9 recommended)

---

## ▶️ Usage

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



## Tests & Lint

Run tests:
```bash
uv run --extra test pytest
```

Run lint:
```bash
uv run --extra lint ruff check .
```

---

## Build

Build sdist + wheel (isolated env):
```bash
uv run --with build python -m build
```

Artifacts land in `dist/`.

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
