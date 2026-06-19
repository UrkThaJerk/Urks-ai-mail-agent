import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_agent import (
    EditOperation,
    TimelineSegment,
    VideoAsset,
    VideoEditingAgent,
    VideoProject,
    build_ffmpeg_command,
    extract_clip,
    filter_top_clips,
    parse_edit_instructions,
    probe_video,
    rank_scenes_by_excitement,
    score_scene_audio_energy,
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

    def test_parse_edit_instructions_extracts_make_clips(self):
        for phrase in ("make clips", "extract clips", "create clips", "split clips"):
            with self.subTest(phrase=phrase):
                operations = parse_edit_instructions(phrase)
                self.assertIn(EditOperation("extract_clips", {}), operations)

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

    @patch("video_agent._run_media_command")
    def test_extract_clip_builds_correct_ffmpeg_command(self, run_media_command):
        run_media_command.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "clip.mp4")
            result = extract_clip("/tmp/input.mp4", 10.0, 30.0, output_path)

        self.assertEqual(output_path, result)
        call_args = run_media_command.call_args[0][0]
        self.assertEqual("ffmpeg", call_args[0])
        self.assertIn("-ss", call_args)
        self.assertIn("10.0", call_args)
        self.assertIn("-t", call_args)
        self.assertIn("20.0", call_args)
        self.assertEqual(output_path, call_args[-1])

    @patch("video_agent._run_media_command")
    def test_extract_clips_from_scenes_creates_one_clip_per_scene(self, run_media_command):
        run_media_command.return_value = MagicMock()

        agent = VideoEditingAgent()
        project = VideoProject(
            source=VideoAsset(path="/tmp/vod.mp4", format="mp4", duration_seconds=60.0),
            scenes=[
                TimelineSegment(start_seconds=0.0, end_seconds=20.0, label="scene_1"),
                TimelineSegment(start_seconds=20.0, end_seconds=45.0, label="scene_2"),
                TimelineSegment(start_seconds=45.0, end_seconds=60.0, label="scene_3"),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_paths = agent.extract_clips_from_scenes(project, output_dir=temp_dir)

        self.assertEqual(3, len(clip_paths))
        self.assertTrue(all("scene_" in p for p in clip_paths))
        self.assertEqual(3, run_media_command.call_count)

    @patch("video_agent.download_video")
    def test_load_from_url_downloads_then_loads_video(self, mock_download):
        agent = VideoEditingAgent()
        fake_project = VideoProject(source=VideoAsset(path="/tmp/2800009020.mp4", format="mp4"))
        mock_download.return_value = "/tmp/2800009020.mp4"

        with patch.object(agent, "load_video", return_value=fake_project) as mock_load:
            result = agent.load_from_url(
                "https://www.twitch.tv/videos/2800009020",
                instructions="make clips",
                download_dir="/tmp",
            )

        mock_download.assert_called_once_with(
            "https://www.twitch.tv/videos/2800009020", output_dir="/tmp", credentials=None
        )
        mock_load.assert_called_once_with("/tmp/2800009020.mp4", "make clips")
        self.assertIs(fake_project, result)

    @patch("video_agent.yt_dlp.YoutubeDL")
    def test_download_video_raises_on_dns_error(self, mock_ydl_cls):
        mock_ydl_cls.return_value.__enter__.return_value.extract_info.side_effect = socket.gaierror(
            -5, "No address associated with hostname"
        )

        from video_agent import VideoEditingError, download_video

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(VideoEditingError) as ctx:
                download_video("https://www.twitch.tv/videos/2800009020", output_dir=tmp)

        self.assertIn("DNS resolution failed", str(ctx.exception))
        self.assertIn("Check your network connection", str(ctx.exception))


class AudioScoringTests(unittest.TestCase):
    @patch("video_agent._run_media_command")
    def test_score_scene_audio_energy_parses_rms(self, mock_run):
        mock_run.return_value = MagicMock(stderr="RMS level dB: -30.0\nRMS level dB: -20.0", stdout="")
        score = score_scene_audio_energy("/tmp/video.mp4", 0.0, 10.0)
        # mean dB = -25 → ((-25 + 60) / 60) = 35/60 ≈ 0.5833
        self.assertAlmostEqual(score, round(35 / 60, 4))

    @patch("video_agent._run_media_command")
    def test_score_scene_audio_energy_clamps_to_zero_on_no_rms(self, mock_run):
        mock_run.return_value = MagicMock(stderr="no audio info", stdout="")
        score = score_scene_audio_energy("/tmp/video.mp4", 0.0, 10.0)
        self.assertEqual(0.0, score)

    def test_score_scene_audio_energy_zero_duration(self):
        score = score_scene_audio_energy("/tmp/video.mp4", 5.0, 5.0)
        self.assertEqual(0.0, score)

    @patch("video_agent._run_media_command")
    def test_score_scene_audio_energy_returns_zero_on_error(self, mock_run):
        from video_agent import VideoEditingError
        mock_run.side_effect = VideoEditingError("ffmpeg failed")
        score = score_scene_audio_energy("/tmp/video.mp4", 0.0, 10.0)
        self.assertEqual(0.0, score)

    @patch("video_agent.score_scene_audio_energy")
    def test_rank_scenes_by_excitement_sorts_descending(self, mock_score):
        mock_score.side_effect = [0.3, 0.9, 0.1]
        asset = VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=30.0)
        scenes = [
            TimelineSegment(0.0, 10.0, "s1"),
            TimelineSegment(10.0, 20.0, "s2"),
            TimelineSegment(20.0, 30.0, "s3"),
        ]
        result = rank_scenes_by_excitement(asset, scenes)
        self.assertEqual(["s2", "s1", "s3"], [s.label for s in result])

    @patch("video_agent.score_scene_audio_energy")
    def test_rank_scenes_marks_above_median_as_key_moment(self, mock_score):
        mock_score.side_effect = [0.2, 0.8, 0.5]
        asset = VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=30.0)
        scenes = [
            TimelineSegment(0.0, 10.0, "s1"),
            TimelineSegment(10.0, 20.0, "s2"),
            TimelineSegment(20.0, 30.0, "s3"),
        ]
        rank_scenes_by_excitement(asset, scenes)
        self.assertFalse(scenes[0].key_moment)  # 0.2 below median 0.5
        self.assertTrue(scenes[1].key_moment)   # 0.8 >= median
        self.assertTrue(scenes[2].key_moment)   # 0.5 == median

    def test_filter_top_clips_returns_n_highest(self):
        scenes = [
            TimelineSegment(0.0, 10.0, "s1", excitement_score=0.1),
            TimelineSegment(10.0, 20.0, "s2", excitement_score=0.9),
            TimelineSegment(20.0, 30.0, "s3", excitement_score=0.5),
            TimelineSegment(30.0, 40.0, "s4", excitement_score=0.7),
        ]
        result = filter_top_clips(scenes, n=2)
        self.assertEqual(2, len(result))
        self.assertEqual("s2", result[0].label)
        self.assertEqual("s4", result[1].label)

    def test_filter_top_clips_n_zero_returns_empty(self):
        scenes = [TimelineSegment(0.0, 10.0, "s1", excitement_score=0.5)]
        self.assertEqual([], filter_top_clips(scenes, n=0))


class ReframeVerticalTests(unittest.TestCase):
    def test_parse_edit_instructions_detects_tiktok_reframe(self):
        for phrase in ("tiktok", "vertical", "9:16", "reframe", "youtube shorts", "reels", "portrait"):
            with self.subTest(phrase=phrase):
                ops = parse_edit_instructions(phrase)
                self.assertIn(EditOperation("reframe_vertical"), ops)

    def test_build_ffmpeg_command_includes_reframe_filter(self):
        project = VideoProject(
            source=VideoAsset(path="/tmp/input.mp4", format="mp4", duration_seconds=10.0),
            operations=[EditOperation("reframe_vertical")],
        )
        command = build_ffmpeg_command(project, "/tmp/out.mp4")
        self.assertIn("-vf", command)
        vf_index = command.index("-vf")
        self.assertIn("1080:1920", command[vf_index + 1])

    @patch("video_agent._run_media_command")
    def test_extract_clips_from_scenes_passes_video_filters_when_reframe(self, mock_run):
        mock_run.return_value = MagicMock()
        agent = VideoEditingAgent()
        project = VideoProject(
            source=VideoAsset(path="/tmp/vod.mp4", format="mp4", duration_seconds=20.0),
            scenes=[TimelineSegment(0.0, 10.0, "s1"), TimelineSegment(10.0, 20.0, "s2")],
            operations=[EditOperation("reframe_vertical")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent.extract_clips_from_scenes(project, output_dir=tmp)

        for call in mock_run.call_args_list:
            cmd = call[0][0]
            self.assertIn("-vf", cmd)
            vf_idx = cmd.index("-vf")
            self.assertIn("1080:1920", cmd[vf_idx + 1])

    @patch("video_agent._run_media_command")
    def test_extract_clips_from_scenes_no_filters_without_reframe(self, mock_run):
        mock_run.return_value = MagicMock()
        agent = VideoEditingAgent()
        project = VideoProject(
            source=VideoAsset(path="/tmp/vod.mp4", format="mp4", duration_seconds=20.0),
            scenes=[TimelineSegment(0.0, 10.0, "s1")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent.extract_clips_from_scenes(project, output_dir=tmp)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("-vf", cmd)

    @patch("video_agent._run_media_command")
    def test_agent_reframe_vertical_appends_operation(self, _mock):
        agent = VideoEditingAgent()
        project = VideoProject(source=VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=10.0))
        agent.reframe_vertical(project)
        self.assertIn(EditOperation("reframe_vertical"), project.operations)

    @patch("video_agent._run_media_command")
    def test_agent_reframe_vertical_idempotent(self, _mock):
        agent = VideoEditingAgent()
        project = VideoProject(source=VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=10.0))
        agent.reframe_vertical(project)
        agent.reframe_vertical(project)
        count = sum(1 for op in project.operations if op.operation == "reframe_vertical")
        self.assertEqual(1, count)


class HighlightInstructionTests(unittest.TestCase):
    def test_parse_edit_instructions_detects_highlight_keywords(self):
        for phrase in ("highlight", "best moments", "best clips", "top moments", "exciting moments"):
            with self.subTest(phrase=phrase):
                ops = parse_edit_instructions(phrase)
                self.assertIn(EditOperation("rank_scenes"), ops)

    def test_parse_edit_instructions_detects_top_n_clips(self):
        ops = parse_edit_instructions("top 3 clips")
        self.assertIn(EditOperation("select_top_clips", {"n": 3}), ops)

    def test_parse_edit_instructions_top_5_fallback(self):
        ops = parse_edit_instructions("highlight reel")
        self.assertIn(EditOperation("select_top_clips", {"n": 5}), ops)

    @patch("video_agent.score_scene_audio_energy", return_value=0.5)
    def test_agent_rank_scenes_auto_detects_if_empty(self, _mock_score):
        agent = VideoEditingAgent()
        asset = VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=30.0)
        project = VideoProject(source=asset)
        with patch("video_agent.detect_scene_segments", return_value=[
            TimelineSegment(0.0, 15.0, "s1"),
            TimelineSegment(15.0, 30.0, "s2"),
        ]) as mock_detect:
            agent.rank_scenes(project)
            mock_detect.assert_called_once_with(asset)
        self.assertEqual(2, len(project.scenes))

    @patch("video_agent.score_scene_audio_energy")
    def test_agent_select_top_clips_limits_scenes(self, mock_score):
        mock_score.side_effect = [0.9, 0.1, 0.5, 0.7, 0.3]
        agent = VideoEditingAgent()
        asset = VideoAsset(path="/tmp/v.mp4", format="mp4", duration_seconds=50.0)
        project = VideoProject(
            source=asset,
            scenes=[TimelineSegment(i * 10.0, (i + 1) * 10.0, f"s{i}") for i in range(5)],
        )
        agent.select_top_clips(project, n=3)
        self.assertEqual(3, len(project.scenes))


if __name__ == "__main__":
    unittest.main()
