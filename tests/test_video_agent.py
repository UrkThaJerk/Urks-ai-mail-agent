import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_agent import (
    EditOperation,
    VideoAsset,
    VideoEditingAgent,
    VideoProject,
    build_ffmpeg_command,
    parse_edit_instructions,
    probe_video,
)


class VideoAgentTests(unittest.TestCase):
    def test_parse_edit_instructions_extracts_expected_operations(self):
        operations = parse_edit_instructions(
            'Detect scenes, apply sepia, add text "Intro", watermark "Urks", sync audio and export to /tmp/output.mp4'
        )

        self.assertIn(EditOperation("detect_scenes", {}), operations)
        self.assertIn(EditOperation("apply_effect", {"name": "sepia"}), operations)
        self.assertIn(EditOperation("sync_audio_video", {}), operations)
        self.assertIn(EditOperation("set_export_path", {"output_path": "/tmp/output.mp4"}), operations)

    @patch("video_agent._run_media_command")
    def test_probe_video_uses_ffprobe_metadata(self, run_media_command):
        run_media_command.return_value.stdout = """
        {
          "format": {"duration": "12.5"},
          "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
            {"codec_type": "audio", "codec_name": "aac"}
          ]
        }
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir, "sample.mp4")
            video_path.touch()
            asset = probe_video(str(video_path))

        self.assertEqual("mp4", asset.format)
        self.assertEqual(12.5, asset.duration_seconds)
        self.assertEqual(1920, asset.width)
        self.assertEqual("h264", asset.video_codec)
        self.assertEqual("aac", asset.audio_codec)

    def test_build_ffmpeg_command_collects_video_and_audio_filters(self):
        project = VideoProject(
            source=VideoAsset(path="/tmp/input.mp4", format="mp4", duration_seconds=10.0),
            instructions='add text "Intro" top-right',
            operations=[
                EditOperation("apply_effect", {"name": "grayscale"}),
                EditOperation("apply_transition", {"name": "fade"}),
                EditOperation("add_text_overlay", {"text": "Intro", "start_seconds": 1.0, "end_seconds": 3.0}),
                EditOperation("add_watermark", {"text": "Urks", "position": "bottom-right"}),
                EditOperation("sync_audio_video", {}),
                EditOperation("mix_audio", {"mode": "equalize", "volume": 1.1}),
            ],
        )

        command = build_ffmpeg_command(project, "/tmp/output.mp4")

        self.assertEqual("ffmpeg", command[0])
        self.assertIn("-vf", command)
        self.assertIn("-af", command)
        self.assertEqual("/tmp/output.mp4", command[-1])
        self.assertTrue(any("hue=s=0" in segment for segment in command))
        self.assertTrue(any("aresample=async=1:first_pts=0" in segment for segment in command))

    def test_process_batch_returns_summaries(self):
        agent = VideoEditingAgent()
        fake_project = VideoProject(source=VideoAsset(path="/tmp/input.mp4", format="mp4"))

        with patch.object(agent, "load_video", return_value=fake_project) as load_video:
            results = agent.process_batch([{"video_path": "/tmp/input.mp4", "instructions": "detect scenes"}])

        load_video.assert_called_once_with("/tmp/input.mp4", "detect scenes")
        self.assertEqual(1, len(results))
        self.assertEqual("/tmp/input.mp4", results[0]["source"]["path"])


if __name__ == "__main__":
    unittest.main()
