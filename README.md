# OpenAI Project Roles Manager

A Streamlit-based desktop tool for **managing custom project roles** in the **OpenAI API Platform**.  
It allows administrators to easily load role definitions from a YAML file, review permissions, and apply those roles to OpenAI projects through the Admin API.

The app supports:

- 🔑 Persistent storage of your Admin API key  
- 📁 Local, browsable, or bundled YAML role configuration files  
- 🌐 Optional GitHub auto-download fallback  
- 🧩 Interactive role selection UI  
- 🚀 One-click creation of multiple project roles  
- 🖥️ Launch via Python API or the command-line (`project-roles`)

## Features

### Role Configuration Handling
- Load `default_project_roles.yaml` from:
  - a **local file** (default path or user-specified)
  - a **CLI-provided path**
  - a **browse dialog** (wxPython)
  - a **GitHub URL fallback**
  - a **bundled resource** included in the package

### Streamlined Admin Workflow
- Review roles and assigned permissions interactively.
- Create multiple roles in bulk using OpenAI Admin API.
- Status feedback, error handling, and full progress reporting.

### Persistent Admin Key
Your Admin key is stored on the local machine (plaintext, for trusted environments) so it loads automatically on each run.

## Installation

### 1. Install from local source
```bash
pip install -e .
```

### 2. Dependencies
Installed automatically:
- streamlit
- requests
- pyyaml
- wxPython
- importlib_resources (Python < 3.9)

## Usage

### CLI Usage
```bash
project-roles
```

With a custom YAML file:
```bash
project-roles --roles-config-path /path/to/default_project_roles.yaml
```

### Python Usage
```python
from openai_project_roles.app import main
main()
```

Custom path:
```python
main(roles_config_path="path/to/roles.yaml")
```

## Role Configuration File Example
```yaml
roles:
  - role_name: "rag-developer"
    description: "Build and maintain RAG pipelines"
    permissions:
      - "rag.read"
      - "rag.write"
```

## App Interface Overview

1. **Credentials**
2. **Role Configuration**
3. **Role Explorer**
4. **Create Roles**

## Security Notice
Admin API keys are stored in plaintext at:
```
~/.openai_admin_key
```
Use only on trusted machines.

## Development
```bash
streamlit run openai_project_roles/app.py
```

Directory structure:
```
openai_project_roles/
├── app.py
├── cli.py
├── default_project_roles.yaml
├── __init__.py
└── pyproject.toml
```

## License
(Insert your license text.)

## Contributions
Pull requests welcome!
