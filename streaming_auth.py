"""streaming_auth.py — authenticate with streaming platforms so that
agents can download subscriber-only VODs and post clips on schedule.

Supported platforms
-------------------
* Twitch   — OAuth 2.0 Device Code flow (no browser required in CI).
             Produces an OAuth token that yt-dlp accepts as a header.
* YouTube  — OAuth 2.0 via google-auth-oauthlib (re-uses the credential
             flow already used by the YouTube uploader in social_agent.py).
             Produces a cookies file + access token for yt-dlp.
* Kick     — Cookie-based (Kick has no public OAuth API).
             Accepts a Netscape-format cookies file exported from a browser.

Token storage
-------------
Credentials are stored in a JSON token store at the path specified by the
STREAMING_TOKEN_STORE env var (default: ~/.urks_streaming_tokens.json).
Each platform keeps its own section; tokens are refreshed automatically
when they expire.

Environment variables
---------------------
STREAMING_TOKEN_STORE   Path to the JSON token store file.

Twitch
  TWITCH_CLIENT_ID      OAuth application client ID.
  TWITCH_CLIENT_SECRET  OAuth application client secret.

YouTube (streaming / download)
  YOUTUBE_CLIENT_SECRETS_FILE   Path to OAuth 2.0 client secrets JSON.
  YOUTUBE_TOKEN_FILE            Path where the OAuth token is persisted
                                (default: youtube_token.json).

Kick
  KICK_COOKIES_FILE     Path to a Netscape-format cookies file for kick.com.

Usage
-----
    from streaming_auth import StreamingCredentials, get_credentials

    creds = get_credentials("twitch")           # authenticate / refresh
    ydl_opts = creds.apply_to_ydl_opts({})      # inject into yt-dlp options

Or run as a CLI to authenticate interactively:

    python streaming_auth.py twitch
    python streaming_auth.py youtube
    python streaming_auth.py kick
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

STREAMING_TOKEN_STORE = os.getenv(
    "STREAMING_TOKEN_STORE",
    str(Path.home() / ".urks_streaming_tokens.json"),
)

TWITCH_AUTH_BASE = "https://id.twitch.tv/oauth2"
TWITCH_SCOPES = ["user:read:email", "clips:edit", "channel:read:stream_key"]


class StreamingAuthError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StreamingCredentials:
    """Holds authentication details for one streaming platform.

    Call :meth:`apply_to_ydl_opts` to inject the credentials into a
    ``yt_dlp.YoutubeDL`` options dict so that authenticated downloads work.
    """

    platform: str
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0          # Unix timestamp; 0 = never expires
    cookies_file: str = ""           # Netscape cookie file path (Kick / YouTube)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def is_expired(self) -> bool:
        if self.expires_at == 0.0:
            return False
        return time.time() >= self.expires_at - 60  # refresh 60 s early

    def apply_to_ydl_opts(self, opts: dict[str, Any]) -> dict[str, Any]:
        """Return *opts* with auth fields injected for this platform."""
        result = dict(opts)
        if self.access_token:
            # yt-dlp accepts OAuth tokens via the http_headers option
            result.setdefault("http_headers", {})
            result["http_headers"]["Authorization"] = f"Bearer {self.access_token}"
        if self.cookies_file and Path(self.cookies_file).exists():
            result["cookiefile"] = self.cookies_file
        return result


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------


def _load_store() -> dict[str, Any]:
    store_path = Path(STREAMING_TOKEN_STORE)
    if store_path.exists():
        try:
            return json.loads(store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            LOGGER.warning("Token store unreadable; starting fresh.")
    return {}


def _save_store(store: dict[str, Any]) -> None:
    store_path = Path(STREAMING_TOKEN_STORE)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    # Restrict permissions to owner-only on POSIX
    try:
        store_path.chmod(0o600)
    except OSError:
        pass


def load_credentials(platform: str) -> StreamingCredentials | None:
    store = _load_store()
    data = store.get(platform)
    if data is None:
        return None
    return StreamingCredentials(
        platform=platform,
        access_token=data.get("access_token", ""),
        refresh_token=data.get("refresh_token", ""),
        expires_at=float(data.get("expires_at", 0.0)),
        cookies_file=data.get("cookies_file", ""),
        extra=data.get("extra", {}),
    )


def save_credentials(creds: StreamingCredentials) -> None:
    store = _load_store()
    store[creds.platform] = {k: v for k, v in asdict(creds).items() if k != "platform"}
    _save_store(store)
    LOGGER.info("Credentials saved for platform '%s'.", creds.platform)


# ---------------------------------------------------------------------------
# Twitch — Device Code flow
# ---------------------------------------------------------------------------


def _twitch_post(endpoint: str, payload: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(
        f"{TWITCH_AUTH_BASE}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise StreamingAuthError(f"Twitch API error {exc.code}: {body}") from exc


def _refresh_twitch(creds: StreamingCredentials) -> StreamingCredentials:
    client_id = os.getenv("TWITCH_CLIENT_ID", creds.extra.get("client_id", ""))
    client_secret = os.getenv("TWITCH_CLIENT_SECRET", creds.extra.get("client_secret", ""))
    if not (client_id and client_secret and creds.refresh_token):
        raise StreamingAuthError(
            "Cannot refresh Twitch token: set TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, "
            "and ensure a refresh_token is stored."
        )
    data = _twitch_post(
        "token",
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": creds.refresh_token,
        },
    )
    creds.access_token = data["access_token"]
    creds.refresh_token = data.get("refresh_token", creds.refresh_token)
    creds.expires_at = time.time() + int(data.get("expires_in", 14400))
    creds.extra["client_id"] = client_id
    creds.extra["client_secret"] = client_secret
    save_credentials(creds)
    LOGGER.info("Twitch token refreshed.")
    return creds


def authenticate_twitch() -> StreamingCredentials:
    """Run the Twitch Device Code flow interactively and return credentials."""
    client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
    client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
    if not client_id:
        raise StreamingAuthError(
            "Set TWITCH_CLIENT_ID (and TWITCH_CLIENT_SECRET) before authenticating with Twitch. "
            "Create an application at https://dev.twitch.tv/console/apps"
        )

    # Step 1: request a device code
    device_data = _twitch_post(
        "device",
        {
            "client_id": client_id,
            "scopes": " ".join(TWITCH_SCOPES),
        },
    )
    device_code = device_data["device_code"]
    user_code = device_data["user_code"]
    verification_uri = device_data.get("verification_uri", "https://www.twitch.tv/activate")
    interval = int(device_data.get("interval", 5))
    expires_in = int(device_data.get("expires_in", 1800))

    print(
        f"\n[Twitch] Visit {verification_uri} and enter code: {user_code}\n"
        f"(expires in {expires_in // 60} minutes)"
    )
    try:
        webbrowser.open(verification_uri)
    except Exception:  # noqa: BLE001
        pass

    # Step 2: poll for the token
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)
        try:
            token_data = _twitch_post(
                "token",
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
        except StreamingAuthError as exc:
            if "authorization_pending" in str(exc):
                continue
            if "slow_down" in str(exc):
                interval += 5
                continue
            raise

        creds = StreamingCredentials(
            platform="twitch",
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", ""),
            expires_at=time.time() + int(token_data.get("expires_in", 14400)),
            extra={"client_id": client_id, "client_secret": client_secret},
        )
        save_credentials(creds)
        LOGGER.info("Twitch authentication successful.")
        return creds

    raise StreamingAuthError("Twitch device code expired before the user authorized.")


# ---------------------------------------------------------------------------
# YouTube — OAuth 2.0 via google-auth-oauthlib
# ---------------------------------------------------------------------------

_YOUTUBE_AUTH_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
]


def authenticate_youtube() -> StreamingCredentials:
    """Run the YouTube OAuth 2.0 installed-app flow and return credentials."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets_file = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "youtube_client_secrets.json")
    token_file = os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")

    goog_creds: Credentials | None = None
    if Path(token_file).exists():
        goog_creds = Credentials.from_authorized_user_file(token_file, _YOUTUBE_AUTH_SCOPES)
    if not goog_creds or not goog_creds.valid:
        if goog_creds and goog_creds.expired and goog_creds.refresh_token:
            goog_creds.refresh(Request())
        else:
            if not Path(secrets_file).exists():
                raise StreamingAuthError(
                    f"YouTube client secrets file not found at '{secrets_file}'. "
                    "Download it from https://console.cloud.google.com/ and set "
                    "YOUTUBE_CLIENT_SECRETS_FILE."
                )
            flow = InstalledAppFlow.from_client_secrets_file(secrets_file, _YOUTUBE_AUTH_SCOPES)
            goog_creds = flow.run_local_server(port=0)
        Path(token_file).write_text(goog_creds.to_json(), encoding="utf-8")

    expires_at = goog_creds.expiry.timestamp() if goog_creds.expiry else 0.0
    creds = StreamingCredentials(
        platform="youtube",
        access_token=goog_creds.token or "",
        refresh_token=goog_creds.refresh_token or "",
        expires_at=expires_at,
        cookies_file="",
        extra={"token_file": token_file},
    )
    save_credentials(creds)
    LOGGER.info("YouTube authentication successful.")
    return creds


def _refresh_youtube(creds: StreamingCredentials) -> StreamingCredentials:
    """Refresh a YouTube access token using the stored token file."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_file = creds.extra.get("token_file") or os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")
    if not Path(token_file).exists():
        raise StreamingAuthError(
            f"YouTube token file not found at '{token_file}'. Re-authenticate with "
            "'python streaming_auth.py youtube'."
        )
    goog_creds = Credentials.from_authorized_user_file(token_file, _YOUTUBE_AUTH_SCOPES)
    if goog_creds.expired and goog_creds.refresh_token:
        goog_creds.refresh(Request())
        Path(token_file).write_text(goog_creds.to_json(), encoding="utf-8")
    creds.access_token = goog_creds.token or ""
    creds.expires_at = goog_creds.expiry.timestamp() if goog_creds.expiry else 0.0
    save_credentials(creds)
    return creds


# ---------------------------------------------------------------------------
# Kick — cookie-based auth
# ---------------------------------------------------------------------------


def authenticate_kick(cookies_file: str | None = None) -> StreamingCredentials:
    """Register a Kick Netscape-format cookies file for yt-dlp downloads.

    Kick does not have a public OAuth API, so authentication is done by
    exporting cookies from a logged-in browser session.  Use a browser
    extension such as "Get cookies.txt LOCALLY" and save the file, then
    pass its path here (or set KICK_COOKIES_FILE).

    Parameters
    ----------
    cookies_file:
        Path to the Netscape-format cookies file.  Falls back to the
        KICK_COOKIES_FILE environment variable.
    """
    path = cookies_file or os.getenv("KICK_COOKIES_FILE", "").strip()
    if not path:
        raise StreamingAuthError(
            "Provide a Netscape-format cookies file via the cookies_file argument "
            "or KICK_COOKIES_FILE environment variable. "
            "Export cookies from a logged-in browser session at https://kick.com using "
            "a browser extension such as 'Get cookies.txt LOCALLY'."
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise StreamingAuthError(
            f"Kick cookies file not found at '{resolved}'. "
            "Export cookies from a logged-in browser session."
        )
    creds = StreamingCredentials(
        platform="kick",
        cookies_file=str(resolved),
    )
    save_credentials(creds)
    LOGGER.info("Kick cookies registered from '%s'.", resolved)
    return creds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_credentials(
    platform: str,
    *,
    force_reauth: bool = False,
    kick_cookies_file: str | None = None,
) -> StreamingCredentials:
    """Return valid credentials for *platform*, refreshing or authenticating
    interactively as needed.

    Parameters
    ----------
    platform:
        One of ``"twitch"``, ``"youtube"``, or ``"kick"``.
    force_reauth:
        When ``True``, ignore any cached token and re-authenticate from scratch.
    kick_cookies_file:
        Path to a Netscape cookies file (only used when platform is ``"kick"``).
    """
    platform = platform.lower().strip()

    if not force_reauth:
        creds = load_credentials(platform)
        if creds is not None:
            if not creds.is_expired():
                return creds
            # Attempt a silent refresh
            try:
                if platform == "twitch":
                    return _refresh_twitch(creds)
                if platform == "youtube":
                    return _refresh_youtube(creds)
                # Kick: cookies don't expire via this API
                return creds
            except StreamingAuthError as exc:
                LOGGER.warning("Silent refresh failed (%s); falling back to interactive auth.", exc)

    if platform == "twitch":
        return authenticate_twitch()
    if platform == "youtube":
        return authenticate_youtube()
    if platform == "kick":
        return authenticate_kick(kick_cookies_file)

    raise StreamingAuthError(
        f"Unknown platform '{platform}'. Supported: twitch, youtube, kick."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> None:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python streaming_auth.py <platform> [--force]")
        print("Platforms: twitch, youtube, kick")
        sys.exit(1)

    platform = args[0].lower()
    force = "--force" in args

    kick_file: str | None = None
    if platform == "kick":
        non_flag = [a for a in args[1:] if not a.startswith("--")]
        kick_file = non_flag[0] if non_flag else None

    creds = get_credentials(platform, force_reauth=force, kick_cookies_file=kick_file)
    print(f"[{platform}] Authenticated successfully.")
    if creds.access_token:
        print(f"  access_token: {creds.access_token[:8]}…")
    if creds.cookies_file:
        print(f"  cookies_file: {creds.cookies_file}")
    print(f"  token_store:  {STREAMING_TOKEN_STORE}")


if __name__ == "__main__":
    _cli()
