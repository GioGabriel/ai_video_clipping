from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.container import AppContainer, build_container
from src.ui.routes import router as ui_router


def create_app(container: AppContainer | None = None, *, start_worker: bool = True) -> FastAPI:
    app_container = container or build_container()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_container.database.initialize()

        if start_worker:
            reconciled_jobs = app_container.job_repository.reconcile_incomplete_jobs(
                "Marked failed during startup because the previous in-memory worker process stopped before completion."
            )
            for job in reconciled_jobs:
                app_container.job_event_repository.create(
                    job_id=job["id"],
                    step=str(job["current_step"]),
                    level="error",
                    message=str(job["error_message"]),
                    created_at=str(job["updated_at"]),
                )
            app_container.job_queue.start()

        app.state.container = app_container
        try:
            yield
        finally:
            if start_worker:
                app_container.job_queue.stop()
            app_container.ollama_runtime.close()
            app_container.viral_detector.close()

    app = FastAPI(title=app_container.settings.app_name, lifespan=lifespan)
    app.include_router(router)
    app.include_router(ui_router)
    app.mount(
        "/assets",
        StaticFiles(directory=app_container.settings.project_root / "frontend" / "assets"),
        name="assets",
    )
    app.mount("/media", StaticFiles(directory=app_container.settings.data_dir), name="media")
    return app


app = create_app()
