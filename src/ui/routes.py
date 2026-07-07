from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)


@router.get("/")
def dashboard(request: Request) -> FileResponse:
    project_root = request.app.state.container.settings.project_root
    return FileResponse(project_root / "frontend" / "index.html")


@router.get("/dashboard")
def dashboard_alias(request: Request) -> FileResponse:
    project_root = request.app.state.container.settings.project_root
    return FileResponse(project_root / "frontend" / "index.html")
