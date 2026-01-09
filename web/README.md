# aidmx Web UI

React + Vite frontend for the aidmx controller, powered by a Flask API.

## Backend (Flask)

Create/activate your Python venv and install deps:

```bash
source env/bin/activate
pip install -r requirements.txt
python -m src.server
```

This starts Flask on http://localhost:5000.

## Frontend (React)

Install node dependencies and run dev server:

```bash
cd web
npm install
npm run dev
```

Open the URL shown by Vite (default http://localhost:5173). The UI polls the Flask API for logs and task status.

## Tabs
- Analyze: Run analyze with options and cancel.
- Play: Play OLA show with options and cancel.
- Universe: View/edit universe JSON (size, groups, fixtures). Save writes to config.json via API.

