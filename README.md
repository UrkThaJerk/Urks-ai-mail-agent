# Urks-ai-mail-agent

Supports both the original mail workflow and a new prototype video editing agent modeled after the same lightweight agent pattern.

## Mail agent

```bash
python agent.py
```

## Video editing agent

Use `URKS_AGENT_TYPE=video` to switch the entrypoint into video editing mode.

```bash
URKS_AGENT_TYPE=video \
VIDEO_INPUT_PATH=/absolute/path/to/input.mp4 \
VIDEO_EDIT_INSTRUCTIONS='detect scenes, apply sepia, add text "Intro", watermark "Urks", sync audio and export to /absolute/path/to/output.mp4' \
python agent.py
```

For batch processing, set `VIDEO_BATCH_SPEC` to a JSON file containing a list of `video_path`, `instructions`, and optional `output_path` objects.