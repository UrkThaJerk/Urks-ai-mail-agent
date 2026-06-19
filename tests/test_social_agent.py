import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from social_agent import (
    ClipMetadata,
    ScheduledPost,
    SocialAgentError,
    SocialMediaAgent,
    SocialPublishResult,
    _default_schedule,
    generate_clip_metadata,
    generate_trending_schedule,
    process_social_jobs,
    slots_to_datetimes,
)


# ---------------------------------------------------------------------------
# Fake OpenAI client helpers
# ---------------------------------------------------------------------------


def _fake_client(responses: list[str]) -> MagicMock:
    """Return a minimal mock that returns successive *responses* from the LLM."""
    calls: list[str] = []

    def _create(**kwargs):
        content = responses[len(calls)]
        calls.append(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# generate_clip_metadata
# ---------------------------------------------------------------------------


class GenerateClipMetadataTests(unittest.TestCase):
    def test_parses_valid_llm_json(self):
        payload = json.dumps(
            {"title": "Epic Clip", "description": "Great moment", "tags": ["gaming", "twitch"]}
        )
        client = _fake_client([payload])
        meta = generate_clip_metadata("/tmp/clips/scene_1.mp4", "gaming", client=client)

        self.assertEqual("Epic Clip", meta.title)
        self.assertEqual("Great moment", meta.description)
        self.assertEqual(["gaming", "twitch"], meta.tags)
        self.assertEqual("/tmp/clips/scene_1.mp4", meta.path)

    def test_falls_back_to_filename_on_invalid_json(self):
        client = _fake_client(["not json at all"])
        meta = generate_clip_metadata("/tmp/clips/scene_2.mp4", "gaming", client=client)

        self.assertEqual("scene_2", meta.title)

    def test_title_is_capped_at_100_chars(self):
        long_title = "A" * 200
        payload = json.dumps({"title": long_title, "description": "", "tags": []})
        client = _fake_client([payload])
        meta = generate_clip_metadata("/tmp/scene.mp4", "sports", client=client)

        self.assertLessEqual(len(meta.title), 100)

    def test_description_is_capped_at_200_chars(self):
        long_desc = "B" * 300
        payload = json.dumps({"title": "T", "description": long_desc, "tags": []})
        client = _fake_client([payload])
        meta = generate_clip_metadata("/tmp/scene.mp4", "sports", client=client)

        self.assertLessEqual(len(meta.description), 200)


# ---------------------------------------------------------------------------
# generate_trending_schedule
# ---------------------------------------------------------------------------


class GenerateTrendingScheduleTests(unittest.TestCase):
    def test_returns_requested_number_of_slots(self):
        slots = [{"platform": "youtube", "day_of_week": "4", "hour_utc": "15"}] * 6
        client = _fake_client([json.dumps(slots)])
        result = generate_trending_schedule(["youtube"], "gaming", 6, client=client)

        self.assertEqual(6, len(result))

    def test_falls_back_to_defaults_on_bad_llm_response(self):
        client = _fake_client(["not json"])
        result = generate_trending_schedule(["youtube", "tiktok"], "gaming", 4, client=client)

        self.assertEqual(4, len(result))
        self.assertTrue(all("platform" in s for s in result))

    def test_falls_back_to_defaults_on_llm_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network down")
        result = generate_trending_schedule(["twitter"], "gaming", 3, client=client)

        self.assertEqual(3, len(result))


class DefaultScheduleTests(unittest.TestCase):
    def test_produces_correct_number_of_slots(self):
        for n in (1, 5, 10):
            with self.subTest(n=n):
                slots = _default_schedule(["youtube", "tiktok"], n)
                self.assertEqual(n, len(slots))

    def test_all_slots_have_required_keys(self):
        slots = _default_schedule(["youtube"], 4)
        for slot in slots:
            self.assertIn("platform", slot)
            self.assertIn("day_of_week", slot)
            self.assertIn("hour_utc", slot)


# ---------------------------------------------------------------------------
# slots_to_datetimes
# ---------------------------------------------------------------------------


class SlotsToDatetimesTests(unittest.TestCase):
    def test_produces_one_datetime_per_slot(self):
        slots = [
            {"platform": "youtube", "day_of_week": "2", "hour_utc": "15"},
            {"platform": "tiktok", "day_of_week": "4", "hour_utc": "13"},
        ]
        ref = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)  # Monday
        result = slots_to_datetimes(slots, reference=ref)

        self.assertEqual(2, len(result))
        # Each result must be in the future relative to ref
        self.assertTrue(all(dt > ref for dt in result))

    def test_datetimes_are_utc(self):
        slots = [{"platform": "youtube", "day_of_week": "1", "hour_utc": "14"}]
        result = slots_to_datetimes(slots)
        self.assertIsNotNone(result[0].tzinfo)


# ---------------------------------------------------------------------------
# SocialMediaAgent
# ---------------------------------------------------------------------------


class SocialMediaAgentTests(unittest.TestCase):
    def _make_agent(self, platforms=None, responses=None):
        responses = responses or []
        client = _fake_client(responses) if responses else MagicMock()
        return SocialMediaAgent(platforms=platforms or ["youtube"], topic="gaming", client=client)

    def test_raises_on_unsupported_platform(self):
        with self.assertRaises(SocialAgentError):
            SocialMediaAgent(platforms=["snapchat"])

    def test_build_schedule_returns_one_post_per_clip_per_platform(self):
        meta_json = json.dumps({"title": "Clip", "description": "desc", "tags": ["a"]})
        schedule_json = json.dumps(
            [{"platform": "youtube", "day_of_week": "4", "hour_utc": "15"}] * 6
        )
        # Two clips × one platform = 2 slots; but we need 2 LLM calls for metadata
        # + 1 for schedule = 3 total calls
        client = _fake_client([schedule_json, meta_json, meta_json])
        agent = SocialMediaAgent(platforms=["youtube"], topic="gaming", client=client)
        ref = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)

        posts = agent.build_schedule(["/tmp/clip1.mp4", "/tmp/clip2.mp4"], reference=ref)

        self.assertEqual(2, len(posts))
        self.assertTrue(all(p.platform == "youtube" for p in posts))
        self.assertTrue(all(p.status == "pending" for p in posts))

    def test_publish_without_upload_sets_all_posts_pending(self):
        meta_json = json.dumps({"title": "T", "description": "D", "tags": []})
        schedule_json = json.dumps(
            [{"platform": "youtube", "day_of_week": "4", "hour_utc": "15"}] * 3
        )
        client = _fake_client([schedule_json, meta_json])
        agent = SocialMediaAgent(platforms=["youtube"], topic="gaming", client=client)

        result = agent.publish(["/tmp/clip1.mp4"], upload_immediately=False)

        self.assertEqual(1, result.total)
        self.assertEqual(0, result.uploaded)
        self.assertEqual(0, result.failed)
        self.assertEqual("pending", result.posts[0].status)

    def test_upload_now_marks_posts_uploaded_on_success(self):
        meta = ClipMetadata(path="/tmp/clip1.mp4", title="T", description="D", tags=[])
        posts = [ScheduledPost(clip=meta, platform="youtube", scheduled_at="2026-06-20T15:00:00+00:00")]

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = "vid123"

        agent = SocialMediaAgent(platforms=["youtube"], topic="gaming", client=MagicMock())
        with patch("social_agent._UPLOADERS", {"youtube": lambda: mock_uploader}):
            result = agent.upload_now(posts)

        self.assertEqual(1, result.uploaded)
        self.assertEqual(0, result.failed)
        self.assertEqual("uploaded", posts[0].status)
        self.assertEqual("vid123", posts[0].platform_id)

    def test_upload_now_marks_posts_failed_on_exception(self):
        meta = ClipMetadata(path="/tmp/clip1.mp4", title="T", description="D", tags=[])
        posts = [ScheduledPost(clip=meta, platform="youtube", scheduled_at="2026-06-20T15:00:00+00:00")]

        mock_uploader = MagicMock()
        mock_uploader.upload.side_effect = RuntimeError("API error")

        agent = SocialMediaAgent(platforms=["youtube"], topic="gaming", client=MagicMock())
        with patch("social_agent._UPLOADERS", {"youtube": lambda: mock_uploader}):
            result = agent.upload_now(posts)

        self.assertEqual(0, result.uploaded)
        self.assertEqual(1, result.failed)
        self.assertEqual("failed", posts[0].status)

    def test_publish_result_summary_is_json_serialisable(self):
        meta = ClipMetadata(path="/tmp/clip.mp4", title="T", description="D", tags=[])
        post = ScheduledPost(clip=meta, platform="tiktok", scheduled_at="2026-06-20T13:00:00+00:00")
        result = SocialPublishResult(total=1, uploaded=0, failed=0, posts=[post])

        summary = result.summary()
        serialised = json.dumps(summary)  # must not raise
        self.assertIn("tiktok", serialised)

    def test_build_schedule_returns_empty_for_no_clips(self):
        agent = SocialMediaAgent(platforms=["youtube"], topic="gaming", client=MagicMock())
        posts = agent.build_schedule([])
        self.assertEqual([], posts)


# ---------------------------------------------------------------------------
# process_social_jobs
# ---------------------------------------------------------------------------


class ProcessSocialJobsTests(unittest.TestCase):
    def test_raises_without_clip_paths(self):
        with patch.dict("os.environ", {"SOCIAL_CLIP_PATHS": ""}, clear=False):
            with self.assertRaises(SocialAgentError):
                process_social_jobs()

    def test_raises_on_invalid_json(self):
        with patch.dict("os.environ", {"SOCIAL_CLIP_PATHS": "not json"}, clear=False):
            with self.assertRaises(SocialAgentError):
                process_social_jobs()

    @patch("social_agent.SocialMediaAgent.publish")
    @patch("builtins.print")
    def test_writes_output_file_and_prints(self, mock_print, mock_publish):
        fake_result = SocialPublishResult(total=1, uploaded=0, failed=0, posts=[])
        mock_publish.return_value = fake_result

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.json"
            with patch.dict(
                "os.environ",
                {
                    "SOCIAL_CLIP_PATHS": '["/tmp/clip.mp4"]',
                    "SOCIAL_TOPIC": "gaming",
                    "SOCIAL_PLATFORMS": "youtube",
                    "SOCIAL_UPLOAD_NOW": "false",
                    "SOCIAL_OUTPUT_PATH": str(output_path),
                },
                clear=False,
            ):
                process_social_jobs()

            self.assertTrue(output_path.exists())
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(1, data["total"])

        mock_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
