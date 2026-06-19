import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streaming_auth import (
    StreamingAuthError,
    StreamingCredentials,
    authenticate_kick,
    get_credentials,
    load_credentials,
    save_credentials,
)


# ---------------------------------------------------------------------------
# StreamingCredentials
# ---------------------------------------------------------------------------


class StreamingCredentialsTests(unittest.TestCase):
    def test_is_expired_returns_false_when_expires_at_is_zero(self):
        creds = StreamingCredentials(platform="twitch", expires_at=0.0)
        self.assertFalse(creds.is_expired())

    def test_is_expired_returns_true_when_past_expiry(self):
        past = time.time() - 10
        creds = StreamingCredentials(platform="twitch", expires_at=past)
        self.assertTrue(creds.is_expired())

    def test_is_expired_returns_false_when_future_expiry(self):
        future = time.time() + 3600
        creds = StreamingCredentials(platform="twitch", expires_at=future)
        self.assertFalse(creds.is_expired())

    def test_apply_to_ydl_opts_injects_authorization_header(self):
        creds = StreamingCredentials(platform="twitch", access_token="tok123")
        opts = creds.apply_to_ydl_opts({})
        self.assertIn("Authorization", opts["http_headers"])
        self.assertIn("tok123", opts["http_headers"]["Authorization"])

    def test_apply_to_ydl_opts_injects_cookiefile(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            cookie_path = f.name
        try:
            creds = StreamingCredentials(platform="kick", cookies_file=cookie_path)
            opts = creds.apply_to_ydl_opts({})
            self.assertEqual(cookie_path, opts["cookiefile"])
        finally:
            Path(cookie_path).unlink(missing_ok=True)

    def test_apply_to_ydl_opts_skips_missing_cookiefile(self):
        creds = StreamingCredentials(platform="kick", cookies_file="/nonexistent/path.txt")
        opts = creds.apply_to_ydl_opts({})
        self.assertNotIn("cookiefile", opts)

    def test_apply_to_ydl_opts_does_not_overwrite_existing_headers(self):
        creds = StreamingCredentials(platform="twitch", access_token="tok456")
        opts = creds.apply_to_ydl_opts({"http_headers": {"X-Custom": "value"}})
        self.assertEqual("value", opts["http_headers"]["X-Custom"])
        self.assertIn("tok456", opts["http_headers"]["Authorization"])

    def test_apply_to_ydl_opts_no_token_leaves_opts_unchanged(self):
        creds = StreamingCredentials(platform="twitch")
        opts = {"format": "best"}
        result = creds.apply_to_ydl_opts(opts)
        self.assertNotIn("http_headers", result)
        self.assertNotIn("cookiefile", result)


# ---------------------------------------------------------------------------
# Token store (load / save)
# ---------------------------------------------------------------------------


class TokenStoreTests(unittest.TestCase):
    def test_round_trip_save_and_load(self):
        creds = StreamingCredentials(
            platform="twitch",
            access_token="abc",
            refresh_token="xyz",
            expires_at=9999999999.0,
            extra={"client_id": "cid"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store_path = str(Path(tmp) / "tokens.json")
            with patch("streaming_auth.STREAMING_TOKEN_STORE", store_path):
                save_credentials(creds)
                loaded = load_credentials("twitch")

        self.assertIsNotNone(loaded)
        self.assertEqual("abc", loaded.access_token)
        self.assertEqual("xyz", loaded.refresh_token)
        self.assertEqual(9999999999.0, loaded.expires_at)
        self.assertEqual("cid", loaded.extra["client_id"])

    def test_load_returns_none_for_unknown_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_path = str(Path(tmp) / "tokens.json")
            with patch("streaming_auth.STREAMING_TOKEN_STORE", store_path):
                result = load_credentials("nonexistent")
        self.assertIsNone(result)

    def test_store_file_gets_restrictive_permissions(self):
        creds = StreamingCredentials(platform="youtube", access_token="t")
        with tempfile.TemporaryDirectory() as tmp:
            store_path = str(Path(tmp) / "tokens.json")
            with patch("streaming_auth.STREAMING_TOKEN_STORE", store_path):
                save_credentials(creds)
            import stat
            mode = Path(store_path).stat().st_mode
            # Owner read+write only (0o600); check group/other have no perms on POSIX
            self.assertEqual(0, mode & stat.S_IRWXG)
            self.assertEqual(0, mode & stat.S_IRWXO)


# ---------------------------------------------------------------------------
# Kick cookie auth
# ---------------------------------------------------------------------------


class KickAuthTests(unittest.TestCase):
    def test_registers_existing_cookies_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            cookie_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmp:
                store_path = str(Path(tmp) / "tokens.json")
                with patch("streaming_auth.STREAMING_TOKEN_STORE", store_path):
                    creds = authenticate_kick(cookies_file=cookie_path)
            self.assertEqual("kick", creds.platform)
            self.assertEqual(cookie_path, creds.cookies_file)
        finally:
            Path(cookie_path).unlink(missing_ok=True)

    def test_raises_when_no_cookies_file_provided(self):
        with patch.dict("os.environ", {"KICK_COOKIES_FILE": ""}, clear=False):
            with self.assertRaises(StreamingAuthError) as ctx:
                authenticate_kick()
        self.assertIn("cookies file", str(ctx.exception).lower())

    def test_raises_when_cookies_file_missing(self):
        with self.assertRaises(StreamingAuthError) as ctx:
            authenticate_kick(cookies_file="/nonexistent/cookies.txt")
        self.assertIn("not found", str(ctx.exception))


# ---------------------------------------------------------------------------
# get_credentials routing
# ---------------------------------------------------------------------------


class GetCredentialsTests(unittest.TestCase):
    def test_returns_cached_valid_credentials(self):
        future = time.time() + 3600
        stored = StreamingCredentials(platform="twitch", access_token="cached", expires_at=future)
        with patch("streaming_auth.load_credentials", return_value=stored):
            result = get_credentials("twitch")
        self.assertEqual("cached", result.access_token)

    def test_raises_for_unknown_platform(self):
        with self.assertRaises(StreamingAuthError) as ctx:
            get_credentials("snapchat")
        self.assertIn("snapchat", str(ctx.exception))

    def test_refreshes_expired_twitch_token(self):
        past = time.time() - 10
        expired = StreamingCredentials(
            platform="twitch",
            access_token="old",
            refresh_token="rtoken",
            expires_at=past,
            extra={"client_id": "cid", "client_secret": "sec"},
        )
        refreshed = StreamingCredentials(
            platform="twitch",
            access_token="new",
            expires_at=time.time() + 3600,
        )
        with patch("streaming_auth.load_credentials", return_value=expired):
            with patch("streaming_auth._refresh_twitch", return_value=refreshed) as mock_refresh:
                result = get_credentials("twitch")
        mock_refresh.assert_called_once()
        self.assertEqual("new", result.access_token)

    def test_falls_back_to_interactive_auth_when_refresh_fails(self):
        past = time.time() - 10
        expired = StreamingCredentials(
            platform="twitch", access_token="old", expires_at=past
        )
        fresh = StreamingCredentials(platform="twitch", access_token="fresh")
        with patch("streaming_auth.load_credentials", return_value=expired):
            with patch("streaming_auth._refresh_twitch", side_effect=StreamingAuthError("fail")):
                with patch("streaming_auth.authenticate_twitch", return_value=fresh) as mock_auth:
                    result = get_credentials("twitch")
        mock_auth.assert_called_once()
        self.assertEqual("fresh", result.access_token)

    def test_force_reauth_skips_cache(self):
        fresh = StreamingCredentials(platform="twitch", access_token="force-fresh")
        with patch("streaming_auth.authenticate_twitch", return_value=fresh) as mock_auth:
            result = get_credentials("twitch", force_reauth=True)
        mock_auth.assert_called_once()
        self.assertEqual("force-fresh", result.access_token)


# ---------------------------------------------------------------------------
# video_agent integration — credentials forwarded to download_video
# ---------------------------------------------------------------------------


class VideoAgentCredentialsTests(unittest.TestCase):
    def test_load_from_url_passes_credentials_to_download_video(self):
        from video_agent import VideoEditingAgent, VideoAsset, VideoProject

        fake_creds = StreamingCredentials(platform="twitch", access_token="tok")
        fake_project = VideoProject(source=VideoAsset(path="/tmp/vod.mp4", format="mp4"))

        with patch("video_agent.download_video", return_value="/tmp/vod.mp4") as mock_dl:
            with patch.object(
                VideoEditingAgent, "load_video", return_value=fake_project
            ):
                agent = VideoEditingAgent()
                agent.load_from_url(
                    "https://www.twitch.tv/videos/2800009020",
                    download_dir="/tmp",
                    credentials=fake_creds,
                )
        mock_dl.assert_called_once_with(
            "https://www.twitch.tv/videos/2800009020",
            output_dir="/tmp",
            credentials=fake_creds,
        )

    def test_download_video_applies_credentials_to_ydl_opts(self):
        """Credentials are applied to yt-dlp options before the download call."""
        import tempfile as _tempfile
        from video_agent import download_video

        creds = StreamingCredentials(platform="twitch", access_token="mytoken")

        captured_opts: dict = {}

        class FakeYDL:
            def __init__(self, opts):
                captured_opts.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def extract_info(self, url, download):
                return {"id": "abc", "ext": "mp4"}

            def prepare_filename(self, info):
                return str(Path(_tempfile.gettempdir()) / "abc.mp4")

        with tempfile.TemporaryDirectory() as tmp:
            # Create the expected output file so the path-check passes
            out = Path(tmp) / "abc.mp4"
            out.touch()

            with patch("video_agent.yt_dlp.YoutubeDL", FakeYDL):
                with patch("video_agent.Path.exists", return_value=True):
                    try:
                        download_video("https://www.twitch.tv/videos/123", output_dir=tmp, credentials=creds)
                    except Exception:
                        pass  # path checks may fail in test; we only care about opts

        self.assertIn("http_headers", captured_opts)
        self.assertIn("mytoken", captured_opts["http_headers"].get("Authorization", ""))


if __name__ == "__main__":
    unittest.main()
