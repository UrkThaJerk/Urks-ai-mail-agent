"""web_app.py — Flask web UI for Urks AI Mail Agent.

Exposes a browser dashboard that lets you trigger any of the four agent modes
(mail, video, social, collective) without touching the terminal.

Usage
-----
    python web_app.py

Then open http://localhost:5000 in your browser.

Environment variables respected
--------------------------------
All the same env vars used by the CLI agents are read at request time, so
you can set them in a .env file or export them in your shell before starting
the server.  The web form values override the environment for the duration of
each request.

Note: Gmail OAuth and YouTube OAuth flows require credentials.json /
token.json files to already exist on disk (same as the CLI).
"""

import io
import json
import os
import traceback

from contextlib import redirect_stdout
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture(func, env_overrides: dict) -> tuple[str, int]:
    """Run *func* with env_overrides applied, capturing stdout.

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
    except Exception:
        return traceback.format_exc(), 500
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


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
    if data.get("video_output_path"):
        env["VIDEO_OUTPUT_PATH"] = data["video_output_path"]

    output, status = _capture(process_video_jobs, env)
    return jsonify({"output": output}), status


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
