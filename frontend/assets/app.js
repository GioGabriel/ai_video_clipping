const JOB_STORAGE_KEY = "ai-clipping-current-job-id";
const VIDEO_STORAGE_KEY = "ai-clipping-current-video-id";
const POLL_INTERVAL_MS = 2000;

const stageDefinitions = [
  {
    key: "queued",
    label: "Queued",
    summary: "Job is stored in SQLite and waiting for an available local worker.",
    liveCopy: "Waiting for the worker queue to pick up the job.",
  },
  {
    key: "download_video",
    label: "Download Source",
    summary: "yt-dlp is pulling the source video into the local data directory.",
    liveCopy: "Downloading the original long-form video locally with yt-dlp.",
  },
  {
    key: "extract_audio",
    label: "Extract Audio",
    summary: "FFmpeg is converting the video soundtrack into a mono 16 kHz WAV file.",
    liveCopy: "Preparing a Whisper-friendly audio track with FFmpeg.",
  },
  {
    key: "transcribe_audio",
    label: "Transcribe Speech",
    summary: "Whisper is generating timestamped transcript segments from the extracted audio.",
    liveCopy: "Running local transcription and building timestamped segments.",
  },
  {
    key: "segment_transcript",
    label: "Segment Sentences",
    summary: "The transcript is being segmented into sentence-aware blocks for clip planning.",
    liveCopy: "Aligning transcript text to sentence boundaries and timing blocks.",
  },
  {
    key: "classify_viral_phrases",
    label: "Classify Phrases",
    summary: "Each sentence is being scanned for viral language patterns like curiosity hooks, advice, and contrarian takes.",
    liveCopy: "Tagging sentences with phrase-level triggers that often perform well on shorts platforms.",
  },
  {
    key: "detect_viral_moments",
    label: "Score Moments",
    summary: "The selected Ollama model is scanning transcript chunks for clip-worthy moments.",
    liveCopy: "Evaluating hooks, opinions, emotional peaks, and story payoffs.",
  },
  {
    key: "detect_hooks",
    label: "Find Hooks",
    summary: "The system is rewinding each candidate to the strongest hook sentence before the moment.",
    liveCopy: "Looking backward for the sharpest opening line before each viral segment.",
  },
  {
    key: "complete_thoughts",
    label: "Finish Thoughts",
    summary: "The pipeline is extending candidate clips until the speaker lands the full thought.",
    liveCopy: "Avoiding mid-sentence endings and waiting for the idea to finish cleanly.",
  },
  {
    key: "optimize_clips",
    label: "Optimize Length",
    summary: "Candidate clips are being expanded or split to hit short-form duration targets.",
    liveCopy: "Targeting 25 to 35 second clips and enforcing the 50 second ceiling.",
  },
  {
    key: "generate_clips",
    label: "Render Clips",
    summary: "FFmpeg is trimming the source and exporting platform-formatted clips.",
    liveCopy: "Cutting candidate moments and framing them for the selected output ratio.",
  },
  {
    key: "generate_subtitles",
    label: "Write Subtitles",
    summary: "Subtitle sidecars are being aligned against each rendered clip.",
    liveCopy: "Generating SRT caption files from overlapping transcript segments.",
  },
  {
    key: "export_assets",
    label: "Export Manifest",
    summary: "The system is writing manifest metadata and final output paths.",
    liveCopy: "Packaging output metadata so the job can be reviewed or extended later.",
  },
  {
    key: "completed",
    label: "Completed",
    summary: "The job finished successfully and clips are ready for review.",
    liveCopy: "Local export finished. Clips and metadata are ready.",
  },
];

const processingSteps = stageDefinitions.filter((step) => step.key !== "completed");
const stageIndexByKey = new Map(stageDefinitions.map((step, index) => [step.key, index]));

const state = {
  currentJobId: null,
  currentVideoId: null,
  pollHandle: null,
};

const elements = {
  submitForm: document.getElementById("submit-form"),
  lookupForm: document.getElementById("lookup-form"),
  videoUrlInput: document.getElementById("video-url"),
  outputAspectRatioSelect: document.getElementById("output-aspect-ratio"),
  ollamaModelSelect: document.getElementById("ollama-model"),
  captionThemeSelect: document.getElementById("caption-theme"),
  jobIdInput: document.getElementById("job-id-input"),
  submitFeedback: document.getElementById("submit-feedback"),
  refreshDashboardButton: document.getElementById("refresh-dashboard"),
  currentJob: document.getElementById("current-job"),
  recentJobs: document.getElementById("recent-jobs"),
  activityFeed: document.getElementById("activity-feed"),
  clipsGrid: document.getElementById("clips-grid"),
  connectionPill: document.getElementById("connection-pill"),
  statQueued: document.getElementById("stat-queued"),
  statRunning: document.getElementById("stat-running"),
  statCompleted: document.getElementById("stat-completed"),
  statFailed: document.getElementById("stat-failed"),
};

const deletableStatuses = new Set(["completed", "failed"]);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDate(value) {
  if (!value) {
    return "n/a";
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatSeconds(value) {
  const totalSeconds = Math.max(Math.round(Number(value) || 0), 0);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "n/a";
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatAspectRatio(value) {
  const normalized = String(value || "9:16");
  const labels = {
    "9:16": "TikTok / Shorts / Reels (9:16)",
    "16:9": "Landscape (16:9)",
    "1:1": "Square (1:1)",
    "4:5": "Instagram Feed (4:5)",
  };
  return labels[normalized] || normalized;
}

function formatModelName(value) {
  return String(value || "llama3").trim() || "llama3";
}

function formatCaptionTheme(value) {
  const normalized = String(value || "tiktok").trim().toLowerCase();
  const labels = {
    tiktok: "TikTok punch",
    cinematic: "Cinematic glow",
    motivational: "Motivational energy",
  };
  return labels[normalized] || normalized;
}

function ensureSelectOption(select, value) {
  if (!select || !value) {
    return;
  }

  const normalized = String(value).trim();
  const hasOption = Array.from(select.options).some((option) => option.value === normalized);
  if (hasOption) {
    return;
  }

  const option = document.createElement("option");
  option.value = normalized;
  option.textContent = normalized;
  select.append(option);
}

function basename(path) {
  if (!path) {
    return "Pending";
  }
  return String(path).split("/").filter(Boolean).pop() || String(path);
}

function setConnectionState(mode, label) {
  elements.connectionPill.textContent = label;
  elements.connectionPill.className = "live-pill";

  if (mode === "ok") {
    return;
  }

  if (mode === "warn") {
    elements.connectionPill.classList.add("status-queued");
    return;
  }

  if (mode === "error") {
    elements.connectionPill.classList.add("status-failed");
  }
}

function stageMeta(stepKey) {
  if (stepKey === "failed") {
    return {
      key: stepKey,
      label: "Failed",
      summary: "The job stopped before the pipeline could complete.",
      liveCopy: "See the error details below to find the failing subsystem.",
    };
  }

  return stageDefinitions.find((step) => step.key === stepKey) || {
    key: stepKey,
    label: humanizeStep(stepKey),
    summary: "Unknown stage.",
    liveCopy: "Stage information unavailable.",
  };
}

function humanizeStep(step) {
  return String(step || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function statusClass(status) {
  return `status-${status}`;
}

function storeState() {
  if (state.currentJobId) {
    localStorage.setItem(JOB_STORAGE_KEY, state.currentJobId);
  } else {
    localStorage.removeItem(JOB_STORAGE_KEY);
  }

  if (state.currentVideoId) {
    localStorage.setItem(VIDEO_STORAGE_KEY, state.currentVideoId);
  } else {
    localStorage.removeItem(VIDEO_STORAGE_KEY);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail || `Request failed with ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function setFeedback(message, isError = false) {
  elements.submitFeedback.textContent = message;
  elements.submitFeedback.style.color = isError ? "var(--bad)" : "var(--muted)";
  elements.submitFeedback.style.borderColor = isError ? "rgba(255, 107, 126, 0.18)" : "rgba(134, 180, 255, 0.12)";
}

function clearPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

function startPolling(jobId) {
  clearPolling();
  state.pollHandle = window.setInterval(() => {
    loadJob(jobId, { silent: true });
  }, POLL_INTERVAL_MS);
}

function progressForJob(job) {
  const currentIndex = stageIndexByKey.get(job.current_step);
  if (job.status === "completed") {
    return 100;
  }
  if (job.status === "failed") {
    if (job.current_step === "failed") {
      return 100;
    }
    return Math.max((((currentIndex ?? 0) + 1) / processingSteps.length) * 100, 8);
  }
  if (currentIndex === undefined || currentIndex < 0) {
    return 6;
  }
  return Math.max((((currentIndex) + 0.55) / processingSteps.length) * 100, 8);
}

function stepProgressLabel(job) {
  const current = Number(job.step_progress_current || 0);
  const total = Number(job.step_progress_total || 0);
  if (total > 0) {
    return `${Math.min(current, total)} / ${total}`;
  }
  if (current > 0) {
    return String(current);
  }
  return "n/a";
}

function renderStepMeter(job) {
  const current = Number(job.step_progress_current || 0);
  const total = Number(job.step_progress_total || 0);
  if (total <= 0) {
    return "";
  }

  const progress = Math.max(Math.min((current / total) * 100, 100), 0);
  const meterState = job.status === "running" ? "running" : "";
  return `
    <div class="job-stage-meter step-meter">
      <div class="meter-labels">
        <span>Current step</span>
        <span>${escapeHtml(stepProgressLabel(job))}</span>
      </div>
      <div class="meter-track compact">
        <div class="meter-fill ${meterState}" style="width: ${progress}%"></div>
      </div>
    </div>
  `;
}

function renderStageCards(job) {
  const currentIndex = stageIndexByKey.get(job.current_step) ?? 0;

  return processingSteps
    .map((step, index) => {
      let stateClass = "pending";
      if (job.status === "failed" && step.key === job.current_step) {
        stateClass = "failed";
      } else if (job.status === "completed" || index < currentIndex) {
        stateClass = "done";
      } else if (
        (job.status === "running" || job.status === "queued") &&
        step.key === job.current_step
      ) {
        stateClass = "active";
      }

      return `
        <article class="stage-card ${stateClass}">
          <div class="stage-card-top">
            <span class="stage-index">${String(index + 1).padStart(2, "0")}</span>
            <span class="stage-status-dot"></span>
          </div>
          <h3 class="stage-name">${escapeHtml(step.label)}</h3>
          <p class="stage-copy">${escapeHtml(step.summary)}</p>
        </article>
      `;
    })
    .join("");
}

function renderCurrentJob(job) {
  if (!job) {
    elements.currentJob.innerHTML = `
      <div class="job-empty">
        <span class="empty-icon">::</span>
        <p>No active job selected. Submit a link or load a previous job.</p>
      </div>
    `;
    elements.activityFeed.className = "activity-feed empty-state";
    elements.activityFeed.textContent = "Select a job to see pipeline events.";
    return;
  }

  const step = stageMeta(job.current_step);
  const progress = progressForJob(job);
  const meterState = job.status === "running" ? "running" : "";
  const links = [];

  if (job.video_id && job.clip_count > 0) {
    links.push(`<button class="secondary-button" type="button" data-load-clips="${escapeHtml(job.video_id)}">Show clips</button>`);
  }
  if (deletableStatuses.has(job.status)) {
    links.push(`<button class="danger-button" type="button" data-delete-job="${escapeHtml(job.id)}">Delete job</button>`);
  }

  const statusLine =
    job.status === "failed"
      ? "The pipeline stopped on the current step."
      : step.liveCopy;
  const activeTask = job.active_task || statusLine;

  const artifacts = [
    {
      label: "Source video",
      status: job.video_path ? "Ready" : "Pending",
      value: basename(job.video_path),
      url: job.video_media_url,
    },
    {
      label: "Audio track",
      status: job.audio_path ? "Ready" : "Pending",
      value: basename(job.audio_path),
      url: job.audio_media_url,
    },
    {
      label: "Transcript",
      status: job.transcript_path ? "Ready" : "Pending",
      value: basename(job.transcript_path),
      url: job.transcript_media_url,
    },
    {
      label: "Manifest",
      status: job.manifest_path ? "Ready" : "Pending",
      value: basename(job.manifest_path),
      url: job.manifest_media_url,
    },
  ];

  elements.currentJob.innerHTML = `
    <div class="job-stage-card">
      <div class="job-stage-hero">
        <div class="job-stage-copy">
          <p class="section-kicker">Video ${escapeHtml(job.video_id)}</p>
          <h3 class="job-title">${escapeHtml(step.label)}</h3>
          <p class="job-subtitle">${escapeHtml(step.summary)}</p>
          <p class="job-live-copy">${escapeHtml(statusLine)}</p>
          <p class="job-live-copy strong">${escapeHtml(activeTask)}</p>
          <p class="job-id">Job ID: ${escapeHtml(job.id)}</p>
        </div>

        <div class="job-status-stack">
          <span class="status-pill ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
          <div class="job-stage-meter">
            <div class="meter-labels">
              <span>Pipeline progress</span>
              <span>${Math.round(progress)}%</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill ${meterState}" style="width: ${progress}%"></div>
            </div>
          </div>
          ${renderStepMeter(job)}
        </div>
      </div>

      <div class="job-stage-grid">
        ${renderStageCards(job)}
      </div>

      <div class="job-metrics-grid">
        <div class="metric-card">
          <span class="metric-label">Live task</span>
          <span class="metric-value metric-value-wrap">${escapeHtml(activeTask)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Source URL</span>
          <span class="metric-value">${escapeHtml(job.source_url)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Created</span>
          <span class="metric-value">${escapeHtml(formatDate(job.created_at))}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Updated</span>
          <span class="metric-value">${escapeHtml(formatDate(job.updated_at))}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Clip Count</span>
          <span class="metric-value">${escapeHtml(job.clip_count)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Step progress</span>
          <span class="metric-value">${escapeHtml(stepProgressLabel(job))}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Output Ratio</span>
          <span class="metric-value">${escapeHtml(formatAspectRatio(job.output_aspect_ratio))}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">AI Model</span>
          <span class="metric-value metric-value-wrap">${escapeHtml(formatModelName(job.ollama_model))}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Caption Vibe</span>
          <span class="metric-value">${escapeHtml(formatCaptionTheme(job.caption_theme))}</span>
        </div>
      </div>

      <div class="artifact-grid">
        ${artifacts
          .map(
            (artifact) => `
              <article class="artifact-card ${artifact.url ? "artifact-ready" : "artifact-pending"}">
                <span class="artifact-label">${escapeHtml(artifact.label)}</span>
                <strong class="artifact-value">${escapeHtml(artifact.value)}</strong>
                <span class="artifact-status">${escapeHtml(artifact.status)}</span>
                ${
                  artifact.url
                    ? `<a class="inline-link" href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer">Open file</a>`
                    : `<span class="artifact-muted">File not created yet</span>`
                }
              </article>
            `,
          )
          .join("")}
      </div>

      ${job.error_message ? `<div class="error-banner">${escapeHtml(job.error_message)}</div>` : ""}

      ${links.length ? `<div class="job-links">${links.join("")}</div>` : ""}
    </div>
  `;

  const loadClipsButton = elements.currentJob.querySelector("[data-load-clips]");
  if (loadClipsButton) {
    loadClipsButton.addEventListener("click", () => loadClips(loadClipsButton.dataset.loadClips));
  }
  const deleteJobButton = elements.currentJob.querySelector("[data-delete-job]");
  if (deleteJobButton) {
    deleteJobButton.addEventListener("click", () => deleteJob(deleteJobButton.dataset.deleteJob));
  }
}

function renderActivityFeed(events) {
  if (!events.length) {
    elements.activityFeed.className = "activity-feed empty-state";
    elements.activityFeed.textContent = "No recorded events for this job yet.";
    return;
  }

  elements.activityFeed.className = "activity-feed";
  elements.activityFeed.innerHTML = [...events]
    .slice(-120)
    .reverse()
    .map((event) => {
      const step = stageMeta(event.step);
      return `
        <article class="activity-item activity-${escapeHtml(event.level)}">
          <div class="activity-marker"></div>
          <div class="activity-body">
            <div class="activity-head">
              <p class="activity-step">${escapeHtml(step.label)}</p>
              <div class="activity-meta">
                <span class="activity-level">${escapeHtml(event.level)}</span>
                <span class="activity-time">${escapeHtml(formatDate(event.created_at))}</span>
              </div>
            </div>
            <p class="activity-message">${escapeHtml(event.message)}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderRecentJobs(jobs) {
  if (!jobs.length) {
    elements.recentJobs.className = "recent-jobs empty-state";
    elements.recentJobs.textContent = "No jobs yet.";
    return;
  }

  elements.recentJobs.className = "recent-jobs";
  elements.recentJobs.innerHTML = jobs
    .map((job) => {
      const step = stageMeta(job.current_step);
      return `
        <article class="recent-job-card">
          <div class="recent-job-head">
            <div class="recent-job-copy">
              <p class="card-kicker">${escapeHtml(job.video_id)}</p>
              <h3>${escapeHtml(step.label)}</h3>
              <p>${escapeHtml(job.source_url)}</p>
              ${
                job.active_task
                  ? `<p class="recent-job-runtime">${escapeHtml(job.active_task)}</p>`
                  : ""
              }
            </div>
            <span class="status-pill ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
          </div>

          <div class="recent-job-meta">
            <span>${escapeHtml(formatDate(job.created_at))}</span>
            <span>${escapeHtml(formatAspectRatio(job.output_aspect_ratio || "9:16"))}</span>
            <span>${escapeHtml(formatCaptionTheme(job.caption_theme || "tiktok"))}</span>
            <span>${escapeHtml(formatModelName(job.ollama_model))}</span>
            <span>${escapeHtml(job.clip_count)} clips</span>
            <span>${escapeHtml(stepProgressLabel(job))}</span>
          </div>
          <div class="recent-job-actions">
            <button class="secondary-button" type="button" data-job-id="${escapeHtml(job.id)}">Open job</button>
            ${
              deletableStatuses.has(job.status)
                ? `<button class="danger-button" type="button" data-delete-job="${escapeHtml(job.id)}">Delete</button>`
                : ""
            }
          </div>
        </article>
      `;
    })
    .join("");

  elements.recentJobs.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.jobIdInput.value = button.dataset.jobId;
      loadJob(button.dataset.jobId);
    });
  });
  elements.recentJobs.querySelectorAll("[data-delete-job]").forEach((button) => {
    button.addEventListener("click", () => deleteJob(button.dataset.deleteJob));
  });
}

function renderClips(videoId, clips) {
  if (!clips.length) {
    elements.clipsGrid.className = "clips-grid empty-state";
    elements.clipsGrid.textContent = videoId
      ? "This job does not have exported clips yet."
      : "Clip exports and output files will appear here.";
    return;
  }

  elements.clipsGrid.className = "clips-grid";
  elements.clipsGrid.innerHTML = clips
    .map(
      (clip) => `
        <article class="clip-card">
          <div class="clip-card-head">
            <div class="clip-copy">
              <p class="card-kicker">Clip ${escapeHtml(clip.sequence_number)}</p>
              <h3>${escapeHtml(clip.hook || `Clip ${clip.sequence_number}`)}</h3>
              <p>${escapeHtml(clip.reason || "Generated clip")}</p>
            </div>
            <div class="clip-file">
              <span class="artifact-label">File</span>
              <strong class="artifact-value">${escapeHtml(basename(clip.file_path))}</strong>
            </div>
          </div>
          <div class="clip-metadata">
            <span class="meta-pill">${escapeHtml(formatSeconds(clip.duration_seconds))}</span>
            <span class="meta-pill">${escapeHtml(formatSeconds(clip.start_seconds))} to ${escapeHtml(formatSeconds(clip.end_seconds))}</span>
            <span class="meta-pill">Score ${escapeHtml(formatNumber(clip.score))}</span>
          </div>
          <div class="clip-actions">
            <a class="inline-link" href="${escapeHtml(clip.media_url)}" target="_blank" rel="noreferrer">Open clip</a>
            ${
              clip.subtitle_url
                ? `<a class="inline-link" href="${escapeHtml(clip.subtitle_url)}" target="_blank" rel="noreferrer">Open subtitles</a>`
                : ""
            }
          </div>
        </article>
      `,
    )
    .join("");
}

function renderStats(jobs) {
  const counts = {
    queued: 0,
    running: 0,
    completed: 0,
    failed: 0,
  };

  jobs.forEach((job) => {
    if (counts[job.status] !== undefined) {
      counts[job.status] += 1;
    }
  });

  elements.statQueued.textContent = counts.queued;
  elements.statRunning.textContent = counts.running;
  elements.statCompleted.textContent = counts.completed;
  elements.statFailed.textContent = counts.failed;
}

async function loadRecentJobs() {
  try {
    const payload = await requestJson("/jobs?limit=8");
    const jobs = payload.jobs || [];
    renderStats(jobs);
    renderRecentJobs(jobs);
    setConnectionState("ok", "Local worker online");
  } catch (error) {
    renderStats([]);
    elements.recentJobs.className = "recent-jobs empty-state";
    elements.recentJobs.textContent = error.message;
    setConnectionState("error", "Backend unreachable");
  }
}

async function loadOllamaModels(selectedModel = null) {
  try {
    const payload = await requestJson("/ollama/models");
    const models = Array.isArray(payload.models) && payload.models.length
      ? payload.models.map((model) => formatModelName(model))
      : [formatModelName(payload.default_model)];
    const fallbackModel = formatModelName(selectedModel || payload.default_model);
    const dedupedModels = [...new Set([...models, fallbackModel])];

    elements.ollamaModelSelect.innerHTML = dedupedModels
      .map(
        (model) =>
          `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`,
      )
      .join("");
    elements.ollamaModelSelect.value = fallbackModel;

    if (!payload.available) {
      setFeedback(
        `Ollama model catalog is unavailable right now. Using ${fallbackModel} until Ollama responds.`,
      );
    }
  } catch (error) {
    const fallbackModel = formatModelName(selectedModel || elements.ollamaModelSelect.value || "llama3");
    elements.ollamaModelSelect.innerHTML = `<option value="${escapeHtml(fallbackModel)}">${escapeHtml(fallbackModel)}</option>`;
    elements.ollamaModelSelect.value = fallbackModel;
  }
}

async function loadClips(videoId) {
  if (!videoId) {
    renderClips(null, []);
    return;
  }

  try {
    const payload = await requestJson(`/clips/${encodeURIComponent(videoId)}`);
    renderClips(videoId, payload.clips || []);
  } catch (error) {
    elements.clipsGrid.className = "clips-grid empty-state";
    elements.clipsGrid.textContent = error.message;
  }
}

async function loadEvents(jobId) {
  if (!jobId) {
    renderActivityFeed([]);
    return;
  }

  try {
    const payload = await requestJson(`/events/${encodeURIComponent(jobId)}`);
    renderActivityFeed(payload.events || []);
  } catch (error) {
    elements.activityFeed.className = "activity-feed empty-state";
    elements.activityFeed.textContent = error.message;
  }
}

async function deleteJob(jobId) {
  if (!jobId) {
    return;
  }

  const confirmed = window.confirm(
    "Delete this job and its local artifacts?\n\nThis removes the job record, clips, transcripts, audio, videos, and related temporary files for that video ID.",
  );
  if (!confirmed) {
    return;
  }

  try {
    const payload = await requestJson(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    const deletedSize = formatBytes(payload.deleted_bytes);
    setFeedback(`Deleted job ${payload.job_id} and reclaimed ${deletedSize}.`);

    if (state.currentJobId === jobId) {
      state.currentJobId = null;
      state.currentVideoId = null;
      storeState();
      renderCurrentJob(null);
      renderClips(null, []);
    }

    await loadRecentJobs();
  } catch (error) {
    setFeedback(error.message, true);
  }
}

async function loadJob(jobId, options = {}) {
  if (!jobId) {
    return;
  }

  const { silent = false } = options;

  try {
    const job = await requestJson(`/status/${encodeURIComponent(jobId)}`);
    state.currentJobId = job.id;
    state.currentVideoId = job.video_id;
    storeState();
    elements.jobIdInput.value = job.id;
    if (job.output_aspect_ratio) {
      elements.outputAspectRatioSelect.value = job.output_aspect_ratio;
    }
    if (job.ollama_model) {
      ensureSelectOption(elements.ollamaModelSelect, job.ollama_model);
      elements.ollamaModelSelect.value = formatModelName(job.ollama_model);
    }
    if (job.caption_theme) {
      elements.captionThemeSelect.value = job.caption_theme;
    }
    renderCurrentJob(job);
    await loadEvents(job.id);

    if (job.status === "queued" || job.status === "running") {
      startPolling(job.id);
      setConnectionState("ok", "Worker active");
      if (!silent) {
        setFeedback(`Tracking ${stageMeta(job.current_step).label.toLowerCase()} for job ${job.id}.`);
      }
    } else {
      clearPolling();
      setConnectionState("ok", "Local worker online");
    }

    if (job.status === "completed" || job.clip_count > 0) {
      await loadClips(job.video_id);
    } else if (job.status === "failed") {
      renderClips(job.video_id, []);
    }
  } catch (error) {
    clearPolling();
    renderCurrentJob(null);
    setConnectionState("error", "Backend unreachable");
    if (!silent) {
      setFeedback(error.message, true);
    }
  } finally {
    await loadRecentJobs();
  }
}

async function submitVideo(event) {
  event.preventDefault();
  const url = elements.videoUrlInput.value.trim();
  const outputAspectRatio = elements.outputAspectRatioSelect.value;
  const ollamaModel = formatModelName(elements.ollamaModelSelect.value);
  const captionTheme = elements.captionThemeSelect.value || "tiktok";
  if (!url) {
    setFeedback("A video URL is required.", true);
    return;
  }

  setFeedback("Submitting job to the local queue...");

  try {
    const payload = await requestJson("/process-video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        output_aspect_ratio: outputAspectRatio,
        caption_theme: captionTheme,
        ollama_model: ollamaModel,
      }),
    });
    state.currentJobId = payload.job_id;
    state.currentVideoId = payload.video_id;
    storeState();
    elements.jobIdInput.value = payload.job_id;
    renderClips(payload.video_id, []);
    setFeedback(
      `Queued job ${payload.job_id} for ${formatAspectRatio(payload.output_aspect_ratio)} with ${formatCaptionTheme(payload.caption_theme)} using ${formatModelName(payload.ollama_model)}.`,
    );
    await loadJob(payload.job_id);
  } catch (error) {
    setFeedback(error.message, true);
  }
}

async function lookupJob(event) {
  event.preventDefault();
  const jobId = elements.jobIdInput.value.trim();
  if (!jobId) {
    setFeedback("Paste a job ID to load it.", true);
    return;
  }
  await loadJob(jobId);
}

async function refreshDashboard() {
  await loadRecentJobs();
  if (state.currentJobId) {
    await loadJob(state.currentJobId, { silent: true });
  }
}

function restoreSavedState() {
  state.currentJobId = localStorage.getItem(JOB_STORAGE_KEY);
  state.currentVideoId = localStorage.getItem(VIDEO_STORAGE_KEY);
  if (state.currentJobId) {
    elements.jobIdInput.value = state.currentJobId;
  }
}

function bindEvents() {
  elements.submitForm.addEventListener("submit", submitVideo);
  elements.lookupForm.addEventListener("submit", lookupJob);
  elements.refreshDashboardButton.addEventListener("click", refreshDashboard);
}

async function initialize() {
  bindEvents();
  restoreSavedState();
  await loadOllamaModels();
  await loadRecentJobs();

  if (state.currentJobId) {
    await loadJob(state.currentJobId, { silent: true });
  }
}

window.addEventListener("beforeunload", clearPolling);
window.addEventListener("DOMContentLoaded", initialize);
