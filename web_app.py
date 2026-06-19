"""web_app.py — Flask web UI for Urks AI Mail Agent.

Exposes a browser dashboard that lets you trigger any of the four agent modes
(mail, video, social, collective) without touching the terminal.  Also provides
file upload and download endpoints so video files can be managed entirely from
the browser.

Usage
-----
    python web_app.py

Then open http://localhost:5000 in your browser.

File storage layout
-------------------
uploads/   — files uploaded through the browser are stored here.
outputs/   — processed/exported files land here (set VIDEO_OUTPUT_PATH to a
             path inside this directory to make exports downloadable).

Environment variables respected
--------------------------------
All the same env vars used by the CLI agents are read at request time, so
you can set them in a .env file or export them in your shell before starting
the server.  The web form values override the environment for the duration of
each request.

WEB_PORT          Port to listen on (default: 5000).
WEB_DEBUG         Enable Flask debug/reload mode (default: false).
WEB_UPLOAD_DIR    Directory for uploaded files (default: uploads/ next to this
                  file).
WEB_OUTPUT_DIR    Directory for processed output files served for download
                  (default: outputs/ next to this file).
WEB_MAX_UPLOAD_MB Maximum upload size in megabytes (default: 500).

Note: Gmail OAuth and YouTube OAuth flows require credentials.json /
token.json files to already exist on disk (same as the CLI).
"""

import io
import json
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.resolve()

UPLOAD_DIR = Path(os.getenv("WEB_UPLOAD_DIR", str(_BASE / "uploads"))).resolve()
OUTPUT_DIR = Path(os.getenv("WEB_OUTPUT_DIR", str(_BASE / "outputs"))).resolve()
MAX_UPLOAD_MB = int(os.getenv("WEB_MAX_UPLOAD_MB", "500"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".flv"}

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Return a filename that contains only safe characters."""
    from werkzeug.utils import secure_filename  # bundled with Flask/Werkzeug
    return secure_filename(name)


def _resolve_download_path(filename: str) -> Path:
    """Resolve *filename* to an absolute path confined to UPLOAD_DIR or OUTPUT_DIR.

    Raises 404 if the file does not exist in either directory.
    """
    safe = _safe_filename(filename)
    if not safe:
        abort(400, description="Invalid filename.")

    for directory in (OUTPUT_DIR, UPLOAD_DIR):
        candidate = (directory / safe).resolve()
        # Guard against path traversal: resolved path must stay inside the directory
        try:
            candidate.relative_to(directory)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    abort(404, description=f"File '{safe}' not found.")


def _capture(func, env_overrides: dict) -> tuple[str, int]:
    """Run *func* with *env_overrides* applied, capturing stdout.

    The raw Python traceback is never returned to callers; only a short error
    message is exposed so that internal details do not leak to the browser.

    Returns (output_text, http_status_code).
    """
    original = {k: os.environ.get(k) for k in env_overrides}
    try:
        for k, v in env_overrides.items():
            os.environ[k] = v

        buf = io.StringIO()
        with redirect_stdout(buf):
            func()
        return buf.getvalue() or "Done.", 200
    except Exception as exc:
        LOGGER.exception("Agent error")
        return f"Error: {type(exc).__name__}: {exc}", 500
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# File management — upload / list / download
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_file():
    """Accept one or more video files and save them to UPLOAD_DIR.

    Returns JSON: {"files": [{"name": ..., "path": ..., "size": ...}, ...]}
    """
    uploaded = request.files.getlist("file")
    if not uploaded:
        abort(400, description="No file(s) provided.")

    saved = []
    for f in uploaded:
        ext = Path(f.filename).suffix.lower() if f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            abort(
                415,
                description=(
                    f"Unsupported file type '{ext}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            )
        safe = _safe_filename(f.filename)
        if not safe:
            abort(400, description="Invalid filename.")

        dest = UPLOAD_DIR / safe
        f.save(str(dest))
        saved.append({"name": safe, "path": str(dest), "size": dest.stat().st_size})

    return jsonify({"files": saved}), 201


@app.route("/files")
def list_files():
    """List all files available in the uploads and outputs directories.

    Returns JSON: {"uploads": [...], "outputs": [...]}
    Each entry has: name, path, size (bytes).
    """
    def _entries(directory: Path) -> list[dict]:
        return sorted(
            [
                {"name": p.name, "path": str(p), "size": p.stat().st_size}
                for p in directory.iterdir()
                if p.is_file()
            ],
            key=lambda e: e["name"],
        )

    return jsonify({"uploads": _entries(UPLOAD_DIR), "outputs": _entries(OUTPUT_DIR)})


@app.route("/download/<path:filename>")
def download_file(filename: str):
    """Serve *filename* from OUTPUT_DIR or UPLOAD_DIR as an attachment."""
    file_path = _resolve_download_path(filename)
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

@app.route("/run/mail", methods=["POST"])
def run_mail():
    from mail_agent import process_emails

    output, status = _capture(process_emails, {})
    return jsonify({"output": output}), status


@app.route("/run/video", methods=["POST"])
def run_video():
    from video_agent import process_video_jobs

    data = request.get_json(silent=True) or {}
    env = {}
    if data.get("video_input_path"):
        env["VIDEO_INPUT_PATH"] = data["video_input_path"]
    if data.get("video_edit_instructions"):
        env["VIDEO_EDIT_INSTRUCTIONS"] = data["video_edit_instructions"]
    # Default the output path into OUTPUT_DIR so exports become downloadable
    output_path = data.get("video_output_path") or str(OUTPUT_DIR / "edited_output.mp4")
    env["VIDEO_OUTPUT_PATH"] = output_path

    output, status = _capture(process_video_jobs, env)
    return jsonify({"output": output, "output_path": output_path}), status


@app.route("/run/social", methods=["POST"])
def run_social():
    from social_agent import process_social_jobs

    data = request.get_json(silent=True) or {}
    env = {}

    # clip_paths can arrive as a JSON array string or a Python list
    clip_paths = data.get("social_clip_paths", "")
    if isinstance(clip_paths, list):
        clip_paths = json.dumps(clip_paths)
    if clip_paths:
        env["SOCIAL_CLIP_PATHS"] = clip_paths

    if data.get("social_topic"):
        env["SOCIAL_TOPIC"] = data["social_topic"]
    if data.get("social_platforms"):
        env["SOCIAL_PLATFORMS"] = data["social_platforms"]
    if data.get("social_upload_now"):
        env["SOCIAL_UPLOAD_NOW"] = data["social_upload_now"]

    output, status = _capture(process_social_jobs, env)
    return jsonify({"output": output}), status


@app.route("/run/collective", methods=["POST"])
def run_collective():
    from collective_agent import process_collective_jobs

    data = request.get_json(silent=True) or {}
    env = {}
    if data.get("collective_objective"):
        env["COLLECTIVE_OBJECTIVE"] = data["collective_objective"]
    if data.get("collective_context"):
        env["COLLECTIVE_CONTEXT"] = data["collective_context"]

    output, status = _capture(process_collective_jobs, env)
    return jsonify({"output": output}), status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", "5000"))
    debug = os.getenv("WEB_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
