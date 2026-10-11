"""One server for everything: the API under /api and the built web app everywhere else.

Run (after `npm run build` in web/): uvicorn api.serve:app --port 8000

In development the Vite server does this job instead (it proxies /api to api.main). This
module is for the one-command demo and the Docker image, where only one process runs.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.main import app as api_app
from api.main import lifespan as api_lifespan
from credit_risk.config import PROJECT_ROOT


def web_dist() -> Path:
    return Path(os.environ.get("GLASSBOX_WEB_DIST", PROJECT_ROOT / "web" / "dist")).resolve()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # A mounted sub-app doesn't get its own startup event, so run the API's (model loading) here.
    async with api_lifespan(api_app):
        yield


app = FastAPI(title="Glassbox", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/api", api_app)  # API docs stay available at /api/docs


@app.get("/{path:path}", include_in_schema=False)
def web(path: str) -> FileResponse:
    """Serve built files as they are; any other path gets index.html, and the React router
    takes it from there (so /assess and /portfolio work on reload)."""
    dist = web_dist()
    if not (dist / "index.html").exists():
        raise HTTPException(503, "Web app not built. Run: cd web && npm run build")
    file = (dist / path).resolve()
    # The resolve + is_relative_to check stops "../" paths from reading files outside dist.
    if path and file.is_file() and file.is_relative_to(dist):
        return FileResponse(file)
    return FileResponse(dist / "index.html")
