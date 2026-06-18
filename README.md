# Urks-ai-mail-agent

Supports the original mail workflow, a prototype video editing agent, and a collaborative three-agent mode where specialists work independently before sharing feedback and learnings.

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

## Collaborative three-agent mode

Use `URKS_AGENT_TYPE=collective` to run three separate specialists:

- a mail specialist
- a video specialist
- a strategy specialist

Each agent works on the objective independently first, then reviews the other agents' work so the group can learn collectively.

```bash
URKS_AGENT_TYPE=collective \
COLLECTIVE_OBJECTIVE='Improve the mail and video workflows together' \
COLLECTIVE_CONTEXT='Keep the agents separate, but let them share feedback after their first pass.' \
python agent.py
```

Set `COLLECTIVE_OUTPUT_PATH` to save the JSON result to a file.