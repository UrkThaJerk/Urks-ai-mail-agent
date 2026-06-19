"""social_agent.py — upload video clips to social media platforms on a
trending-aware schedule.

Supported platforms
-------------------
* YouTube  — uses the YouTube Data API v3 via google-api-python-client
* TikTok   — stub using the TikTok Content Posting API (OAuth 2.0)
* Twitter/X — uses the v2 media-upload + tweet endpoints via tweepy

Trending schedule
-----------------
The agent asks an LLM (OpenAI) for the best days/times to post for each
platform given a topic/niche, then sorts the provided clips by scene
importance and assigns each one a scheduled upload slot.

Environment variables
---------------------
OPENAI_API_KEY           Required for trending schedule generation.
SOCIAL_MODEL             LLM model to use (default: gpt-4o-mini).

YouTube
  YOUTUBE_CLIENT_SECRETS_FILE   Path to OAuth 2.0 client secrets JSON.
  YOUTUBE_TOKEN_FILE            Path where the OAuth token is persisted
                                (default: youtube_token.json).

TikTok
  TIKTOK_ACCESS_TOKEN           TikTok Content Posting API access token.

Twitter/X
  TWITTER_BEARER_TOKEN          Twitter API v2 bearer token.
  TWITTER_API_KEY               Twitter API key (consumer key).
  TWITTER_API_SECRET            Twitter API secret (consumer secret).
  TWITTER_ACCESS_TOKEN          Twitter user access token.
  TWITTER_ACCESS_SECRET         Twitter user access token secret.

Entrypoint
----------
Set SOCIAL_CLIP_PATHS to a JSON array of clip file paths (and optionally
SOCIAL_TOPIC to describe the niche, e.g. "gaming highlights"), then run:

    URKS_AGENT_TYPE=social python agent.py

Or call process_social_jobs() directly.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

SOCIAL_MODEL = os.getenv("SOCIAL_MODEL", "gpt-4o-mini")

SUPPORTED_PLATFORMS = {"youtube", "tiktok", "twitter"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ClipMetadata:
    path: str
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ScheduledPost:
    clip: ClipMetadata
    platform: str
    scheduled_at: str  # ISO-8601 UTC
    status: str = "pending"  # pending | uploaded | failed
    platform_id: str = ""  # returned by the platform on success


@dataclass
class SocialPublishResult:
    total: int
    uploaded: int
    failed: int
    posts: list[ScheduledPost] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "uploaded": self.uploaded,
            "failed": self.failed,
            "posts": [asdict(post) for post in self.posts],
        }


class SocialAgentError(Exception):
    pass


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _get_openai_client() -> Any:
    from openai import OpenAI  # lazy import keeps module importable without openai

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_clip_metadata(clip_path: str, topic: str, client: Any | None = None) -> ClipMetadata:
    """Ask the LLM to produce a title, description, and tags for a clip."""
    llm = client or _get_openai_client()
    filename = Path(clip_path).name
    response = llm.chat.completions.create(
        model=SOCIAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a social media content strategist. "
                    "Given a video clip filename and topic, produce a catchy title (max 100 chars), "
                    "a short description (max 200 chars), and 5 relevant hashtag-style tags. "
                    "Reply ONLY with a JSON object with keys: title, description, tags (list of strings)."
                ),
            },
            {
                "role": "user",
                "content": f"Filename: {filename}\nTopic/niche: {topic or 'gaming highlights'}",
            },
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback: use filename as title
        LOGGER.warning("LLM returned non-JSON metadata for %s; using filename fallback.", filename)
        data = {"title": Path(clip_path).stem, "description": "", "tags": []}

    return ClipMetadata(
        path=clip_path,
        title=str(data.get("title", Path(clip_path).stem))[:100],
        description=str(data.get("description", ""))[:200],
        tags=[str(t) for t in data.get("tags", [])],
    )


def generate_trending_schedule(
    platforms: list[str],
    topic: str,
    num_slots: int,
    client: Any | None = None,
) -> list[dict[str, str]]:
    """Return a list of {platform, day_of_week, hour_utc} dicts ordered by
    expected reach, as suggested by the LLM for the given topic/niche.

    Falls back to hardcoded best-practice defaults if the LLM is unavailable
    or returns unparseable output.
    """
    llm = client or _get_openai_client()
    platform_list = ", ".join(platforms)
    prompt = (
        f"Platforms: {platform_list}\n"
        f"Topic/niche: {topic or 'gaming highlights'}\n"
        f"How many upload slots needed: {num_slots}\n\n"
        "Return a JSON array of objects, each with keys: "
        "'platform' (one of the platforms above), "
        "'day_of_week' (0=Monday … 6=Sunday), "
        "'hour_utc' (0-23). "
        f"Include exactly {num_slots} entries, ordered from highest expected reach to lowest. "
        "Use real trending data patterns. Reply ONLY with the JSON array."
    )
    try:
        response = llm.chat.completions.create(
            model=SOCIAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media scheduling expert. "
                        "You know the best days and times to post on each platform for maximum reach."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        slots = json.loads(raw)
        if isinstance(slots, list) and len(slots) >= num_slots:
            return slots[:num_slots]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Trending schedule LLM call failed (%s); using defaults.", exc)

    return _default_schedule(platforms, num_slots)


def _default_schedule(platforms: list[str], num_slots: int) -> list[dict[str, str]]:
    """Research-backed fallback schedule when LLM is unavailable."""
    defaults: list[dict[str, str]] = []
    # Best-practice posting windows per platform
    platform_windows: dict[str, list[tuple[int, int]]] = {
        "youtube": [(4, 15), (5, 18), (6, 14), (2, 17)],  # (day, hour UTC)
        "tiktok": [(2, 13), (4, 15), (1, 10), (5, 20)],
        "twitter": [(1, 14), (3, 17), (4, 12), (2, 15)],
    }
    idx = 0
    while len(defaults) < num_slots:
        for platform in platforms:
            windows = platform_windows.get(platform, [(1, 14)])
            day, hour = windows[idx % len(windows)]
            defaults.append({"platform": platform, "day_of_week": str(day), "hour_utc": str(hour)})
            if len(defaults) >= num_slots:
                break
        idx += 1
    return defaults[:num_slots]


def slots_to_datetimes(slots: list[dict[str, str]], reference: datetime | None = None) -> list[datetime]:
    """Convert day_of_week / hour_utc slots to concrete UTC datetimes
    starting from the next occurrence of each slot after *reference*.
    """
    ref = reference or datetime.now(tz=timezone.utc)
    result: list[datetime] = []
    for slot in slots:
        target_dow = int(slot.get("day_of_week", 1)) % 7
        target_hour = int(slot.get("hour_utc", 14)) % 24
        days_ahead = (target_dow - ref.weekday()) % 7 or 7
        target = ref.replace(hour=target_hour, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        result.append(target)
    return result


# ---------------------------------------------------------------------------
# Platform uploaders
# ---------------------------------------------------------------------------


class YouTubeUploader:
    """Upload a video to YouTube using the Data API v3."""

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

    def __init__(self) -> None:
        self._service: Any | None = None

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        secrets_file = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "youtube_client_secrets.json")
        token_file = os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")

        creds: Credentials | None = None
        if Path(token_file).exists():
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(secrets_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            Path(token_file).write_text(creds.to_json(), encoding="utf-8")

        self._service = build("youtube", "v3", credentials=creds)
        return self._service

    def upload(self, clip: ClipMetadata) -> str:
        """Upload *clip* and return the YouTube video ID."""
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        body = {
            "snippet": {
                "title": clip.title or Path(clip.path).stem,
                "description": clip.description,
                "tags": clip.tags,
                "categoryId": "20",  # Gaming
            },
            "status": {"privacyStatus": "public"},
        }
        media = MediaFileUpload(clip.path, chunksize=-1, resumable=True)
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()
        video_id: str = response.get("id", "")
        LOGGER.info("YouTube upload complete: https://youtu.be/%s", video_id)
        return video_id


class TikTokUploader:
    """Upload a video to TikTok using the Content Posting API."""

    _API_INIT = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    _API_CHECK = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"

    def __init__(self) -> None:
        self._token = os.getenv("TIKTOK_ACCESS_TOKEN", "")

    def upload(self, clip: ClipMetadata) -> str:
        """Upload *clip* and return the TikTok publish_id."""
        import urllib.request

        if not self._token:
            raise SocialAgentError(
                "TIKTOK_ACCESS_TOKEN is not set. "
                "Obtain an access token from the TikTok developer portal."
            )

        clip_path = Path(clip.path)
        file_size = clip_path.stat().st_size

        # Step 1: initialise upload
        init_payload = json.dumps(
            {
                "post_info": {
                    "title": clip.title or clip_path.stem,
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": file_size,
                    "chunk_size": file_size,
                    "total_chunk_count": 1,
                },
            }
        ).encode()
        req = urllib.request.Request(
            self._API_INIT,
            data=init_payload,
            headers={
                "Authorization": f"******",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )
        with urllib.request.urlopen(req) as resp:
            init_data = json.loads(resp.read().decode())

        if init_data.get("error", {}).get("code") != "ok":
            raise SocialAgentError(f"TikTok init failed: {init_data}")

        upload_url: str = init_data["data"]["upload_url"]
        publish_id: str = init_data["data"]["publish_id"]

        # Step 2: upload the binary chunk
        video_bytes = clip_path.read_bytes()
        upload_req = urllib.request.Request(
            upload_url,
            data=video_bytes,
            headers={
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                "Content-Length": str(file_size),
            },
            method="PUT",
        )
        with urllib.request.urlopen(upload_req):
            pass

        LOGGER.info("TikTok upload complete: publish_id=%s", publish_id)
        return publish_id


class TwitterUploader:
    """Post a video clip to Twitter/X using the v2 API via tweepy."""

    def __init__(self) -> None:
        self._bearer = os.getenv("TWITTER_BEARER_TOKEN", "")
        self._api_key = os.getenv("TWITTER_API_KEY", "")
        self._api_secret = os.getenv("TWITTER_API_SECRET", "")
        self._access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self._access_secret = os.getenv("TWITTER_ACCESS_SECRET", "")

    def _get_client(self) -> Any:
        import tweepy  # lazy import

        if not all([self._api_key, self._api_secret, self._access_token, self._access_secret]):
            raise SocialAgentError(
                "Twitter credentials incomplete. Set TWITTER_API_KEY, TWITTER_API_SECRET, "
                "TWITTER_ACCESS_TOKEN, and TWITTER_ACCESS_SECRET."
            )
        return tweepy.Client(
            bearer_token=self._bearer,
            consumer_key=self._api_key,
            consumer_secret=self._api_secret,
            access_token=self._access_token,
            access_token_secret=self._access_secret,
        )

    def _upload_media(self, clip_path: str) -> str:
        """Upload media via the v1.1 chunked upload endpoint; return media_id."""
        import tweepy

        auth = tweepy.OAuth1UserHandler(
            self._api_key, self._api_secret, self._access_token, self._access_secret
        )
        api_v1 = tweepy.API(auth)
        media = api_v1.media_upload(
            filename=clip_path,
            media_category="tweet_video",
            chunked=True,
        )
        return str(media.media_id_string)

    def upload(self, clip: ClipMetadata) -> str:
        """Upload *clip* and return the tweet ID."""
        client = self._get_client()
        media_id = self._upload_media(clip.path)
        tweet_text = clip.title or Path(clip.path).stem
        if clip.tags:
            hashtags = " ".join(f"#{t.lstrip('#')}" for t in clip.tags[:3])
            tweet_text = f"{tweet_text}\n{hashtags}"
        response = client.create_tweet(text=tweet_text[:280], media_ids=[media_id])
        tweet_id: str = str(response.data["id"])
        LOGGER.info("Twitter upload complete: tweet_id=%s", tweet_id)
        return tweet_id


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


_UPLOADERS: dict[str, type] = {
    "youtube": YouTubeUploader,
    "tiktok": TikTokUploader,
    "twitter": TwitterUploader,
}


class SocialMediaAgent:
    """Schedules and uploads video clips to social media platforms.

    Parameters
    ----------
    platforms:
        Which platforms to publish to (default: all three).
    topic:
        The content niche/topic, used to generate trending schedules and
        clip metadata via the LLM (default: "gaming highlights").
    client:
        An OpenAI-compatible client; injected in tests.
    """

    def __init__(
        self,
        platforms: list[str] | None = None,
        topic: str = "gaming highlights",
        client: Any | None = None,
    ) -> None:
        self.platforms = [p.lower() for p in (platforms or list(SUPPORTED_PLATFORMS))]
        invalid = set(self.platforms) - SUPPORTED_PLATFORMS
        if invalid:
            raise SocialAgentError(f"Unsupported platforms: {invalid}. Choose from {SUPPORTED_PLATFORMS}.")
        self.topic = topic
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_schedule(
        self,
        clip_paths: list[str],
        reference: datetime | None = None,
    ) -> list[ScheduledPost]:
        """Assign each clip a trending upload slot and generate its metadata.

        Returns a list of :class:`ScheduledPost` objects ordered by
        scheduled time.
        """
        if not clip_paths:
            return []

        num_slots = len(clip_paths) * len(self.platforms)
        raw_slots = generate_trending_schedule(self.platforms, self.topic, num_slots, self._client)
        scheduled_times = slots_to_datetimes(raw_slots, reference=reference)

        posts: list[ScheduledPost] = []
        slot_index = 0
        for clip_path in clip_paths:
            meta = generate_clip_metadata(clip_path, self.topic, client=self._client)
            for platform in self.platforms:
                slot_dt = scheduled_times[slot_index % len(scheduled_times)]
                posts.append(
                    ScheduledPost(
                        clip=meta,
                        platform=platform,
                        scheduled_at=slot_dt.isoformat(),
                    )
                )
                slot_index += 1

        posts.sort(key=lambda p: p.scheduled_at)
        return posts

    def upload_now(self, posts: list[ScheduledPost]) -> SocialPublishResult:
        """Upload all *posts* immediately, regardless of scheduled time.

        Marks each post's status as ``uploaded`` or ``failed``.
        """
        uploaded = 0
        failed = 0
        for post in posts:
            uploader_cls = _UPLOADERS.get(post.platform)
            if uploader_cls is None:
                LOGGER.error("No uploader for platform '%s'.", post.platform)
                post.status = "failed"
                failed += 1
                continue
            try:
                uploader = uploader_cls()
                post.platform_id = uploader.upload(post.clip)
                post.status = "uploaded"
                uploaded += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Upload failed for %s on %s: %s", post.clip.path, post.platform, exc)
                post.status = "failed"
                failed += 1

        return SocialPublishResult(
            total=len(posts),
            uploaded=uploaded,
            failed=failed,
            posts=posts,
        )

    def publish(
        self,
        clip_paths: list[str],
        upload_immediately: bool = False,
        reference: datetime | None = None,
    ) -> SocialPublishResult:
        """Build a trending schedule for *clip_paths* and optionally upload.

        Parameters
        ----------
        clip_paths:
            Local paths to video clip files.
        upload_immediately:
            When ``True``, upload all clips right now instead of waiting for
            the scheduled time.  Useful for CI/CD pipelines or one-shot runs.
        reference:
            Anchor datetime for slot computation (default: now UTC).
        """
        posts = self.build_schedule(clip_paths, reference=reference)
        if upload_immediately:
            return self.upload_now(posts)
        # Return the schedule without uploading
        return SocialPublishResult(total=len(posts), uploaded=0, failed=0, posts=posts)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def process_social_jobs() -> None:
    """Read configuration from environment variables and run the agent."""
    raw_paths = os.getenv("SOCIAL_CLIP_PATHS", "").strip()
    if not raw_paths:
        raise SocialAgentError(
            "Set SOCIAL_CLIP_PATHS to a JSON array of clip file paths, e.g.: "
            'SOCIAL_CLIP_PATHS=\'["/tmp/clips/clip1.mp4","/tmp/clips/clip2.mp4"]\''
        )

    try:
        clip_paths: list[str] = json.loads(raw_paths)
    except json.JSONDecodeError as exc:
        raise SocialAgentError(f"SOCIAL_CLIP_PATHS must be a valid JSON array: {exc}") from exc

    raw_platforms = os.getenv("SOCIAL_PLATFORMS", "").strip()
    platforms = [p.strip() for p in raw_platforms.split(",")] if raw_platforms else None

    topic = os.getenv("SOCIAL_TOPIC", "gaming highlights").strip()
    upload_now = os.getenv("SOCIAL_UPLOAD_NOW", "").strip().lower() in ("1", "true", "yes")
    output_path = os.getenv("SOCIAL_OUTPUT_PATH", "").strip()

    agent = SocialMediaAgent(platforms=platforms, topic=topic)
    result = agent.publish(clip_paths, upload_immediately=upload_now)

    output = json.dumps(result.summary(), indent=2)
    if output_path:
        resolved = Path(output_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(output, encoding="utf-8")

    print(output)


if __name__ == "__main__":
    process_social_jobs()
