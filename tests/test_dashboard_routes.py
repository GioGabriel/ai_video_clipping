from fastapi.testclient import TestClient

def test_dashboard_page_is_served(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "AI Clipping System" in response.text
    assert "/assets/app.js" in response.text


def test_recent_jobs_endpoint_returns_collection(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        response = client.get("/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert "jobs" in payload
    assert isinstance(payload["jobs"], list)


def test_job_events_endpoint_returns_collection(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        create_response = client.post("/process-video", json={"url": "https://example.com/video"})
        job_id = create_response.json()["job_id"]
        response = client.get(f"/events/{job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert isinstance(payload["events"], list)


def test_status_endpoint_returns_runtime_progress_fields(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        create_response = client.post("/process-video", json={"url": "https://example.com/video"})
        job_id = create_response.json()["job_id"]
        response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["step_progress_current"] == 0
    assert payload["step_progress_total"] == 0
    assert isinstance(payload["active_task"], str)
    assert isinstance(payload["ollama_model"], str)
    assert payload["caption_theme"] == "tiktok"


def test_process_video_accepts_output_aspect_ratio_and_model(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        response = client.post(
            "/process-video",
            json={
                "url": "https://example.com/video",
                "output_aspect_ratio": "16:9",
                "caption_theme": "motivational",
                "ollama_model": "qwen2.5:7b",
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["output_aspect_ratio"] == "16:9"
    assert payload["caption_theme"] == "motivational"
    assert payload["ollama_model"] == "qwen2.5:7b"


def test_ollama_models_endpoint_returns_catalog(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        response = client.get("/ollama/models")

    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert isinstance(payload["models"], list)
    assert isinstance(payload["default_model"], str)
    assert isinstance(payload["available"], bool)
    assert any(str(model).startswith("deepseek-r1") for model in payload["models"])
    assert any(str(model).startswith("qwen2.5") for model in payload["models"])


def test_jobs_endpoints_normalize_legacy_output_aspect_ratio_strings(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        container = client.app.state.container
        job_id = "legacy-aspect-job"
        video_id = "legacyaspect01"

        container.job_repository.create(job_id=job_id, video_id=video_id, source_url="https://example.com/video")
        container.job_repository.update(
            job_id,
            output_aspect_ratio="OutputAspectRatio.LANDSCAPE_16_9",
            status="failed",
            current_step="download_video",
        )

        jobs_response = client.get("/jobs")
        status_response = client.get(f"/status/{job_id}")

        assert jobs_response.status_code == 200
        assert status_response.status_code == 200
        assert any(job["id"] == job_id and job["output_aspect_ratio"] == "16:9" for job in jobs_response.json()["jobs"])
        assert status_response.json()["output_aspect_ratio"] == "16:9"

        container.job_event_repository.delete_by_job_id(job_id)
        container.clip_repository.delete_by_job_id(job_id)
        container.job_repository.delete(job_id)


def test_delete_job_endpoint_removes_terminal_job_and_artifacts(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        container = client.app.state.container
        job_id = "cleanup-job-test"
        video_id = "cleanupvid01"
        job_repository = container.job_repository
        clip_repository = container.clip_repository
        job_event_repository = container.job_event_repository

        video_path = container.settings.videos_dir / f"{video_id}.mp4"
        partial_path = container.settings.videos_dir / f"{video_id}.mp4.part"
        audio_path = container.settings.audio_dir / f"{video_id}.wav"
        transcript_path = container.settings.transcripts_dir / f"{video_id}.json"
        clip_dir = container.settings.clips_dir / video_id
        global_manifest_path = container.settings.clips_dir / "manifest.json"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_dir / "clip_001.mp4"
        subtitle_path = clip_dir / "subtitles" / "clip_001.srt"
        subtitle_path.parent.mkdir(parents=True, exist_ok=True)

        for path in [video_path, partial_path, audio_path, transcript_path, clip_path, subtitle_path, global_manifest_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test", encoding="utf-8")

        job_repository.create(job_id=job_id, video_id=video_id, source_url="https://example.com/video")
        job_repository.update(
            job_id,
            status="failed",
            current_step="download_video",
            video_path=str(video_path),
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
            manifest_path=str(clip_dir / "manifest.json"),
            error_message="download failed",
        )
        (clip_dir / "manifest.json").write_text("{}", encoding="utf-8")
        clip_repository.replace_for_job(
            job_id,
            video_id,
            [],
        )
        job_event_repository.create(job_id=job_id, step="download_video", level="error", message="download failed")

        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == job_id
        assert payload["deleted_bytes"] > 0
        assert job_repository.get(job_id) is None
        assert job_event_repository.list_by_job_id(job_id) == []
        assert not video_path.exists()
        assert not partial_path.exists()
        assert not audio_path.exists()
        assert not transcript_path.exists()
        assert not clip_dir.exists()
        assert not global_manifest_path.exists()


def test_delete_job_endpoint_rejects_active_jobs(dashboard_app) -> None:
    with TestClient(dashboard_app) as client:
        container = client.app.state.container
        job_id = "active-cleanup-job-test"
        video_id = "activevid01"

        container.job_repository.create(job_id=job_id, video_id=video_id, source_url="https://example.com/video")
        container.job_repository.update(job_id, status="running", current_step="download_video")

        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409
        assert "Active jobs cannot be deleted yet" in response.json()["detail"]

        container.job_event_repository.delete_by_job_id(job_id)
        container.clip_repository.delete_by_job_id(job_id)
        container.job_repository.delete(job_id)
