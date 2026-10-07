"""API server entrypoint. Run with: python -m app.web

Serves http://127.0.0.1:8000 (interactive docs at /docs). The Streamlit UI
runs separately: streamlit run app/ui.py (port 8501).
"""

import uvicorn

from app.backend import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)