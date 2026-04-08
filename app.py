"""
FastAPI server for the Medication Dosing OpenEnv environment.

This is an alias for server.app — the canonical entrypoint.
The openenv.yaml points to server.app:app, so this file
re-exports for convenience if anyone runs `uvicorn app:app`.
"""

from server.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
