import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from app.log import get_logger

logger = get_logger(__name__)

load_dotenv()  # reads .env into environment variables

# attempts=1 disables the SDK's built-in retries: our retry loop below is the
# only retry layer, so one logical call never becomes a burst of HTTP requests.
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

RETRYABLE = {429, 500, 503, 504}  # temporary errors worth retrying
MAX_ATTEMPTS = 5

QUOTA_DELAY_RE = re.compile(r"retry in (\d+)h")

MIN_INTERVAL = 7.0  # seconds between calls: the free tier rejects bursts (~10+ RPM)
_last_call = 0.0


def _daily_quota_hours(e) -> int:
    """Hours until a daily quota resets, or 0 if this isn't a daily-quota error."""
    match = QUOTA_DELAY_RE.search(str(e))
    return int(match.group(1)) if match else 0


def generate(**kwargs):
    """Call Gemini, pacing requests and retrying temporary errors."""
    global _last_call
    for attempt in range(MAX_ATTEMPTS):
        gap = time.monotonic() - _last_call
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        _last_call = time.monotonic()
        try:
            response = client.models.generate_content(**kwargs)
            _log_usage(response)
            return response
        except errors.APIError as e:
            is_last = attempt == MAX_ATTEMPTS - 1
            if e.code not in RETRYABLE or is_last:
                raise  # not temporary, or out of attempts
            if e.code == 429:
                hours = _daily_quota_hours(e)
                if hours:
                    logger.error(
                        "daily quota exhausted, resets in ~%sh — not retrying", hours
                    )
                    raise
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            logger.warning(
                "Gemini error %s, retrying in %ss (attempt %s/%s)",
                e.code,
                wait,
                attempt + 1,
                MAX_ATTEMPTS,
            )
            time.sleep(wait)


def _log_usage(response) -> None:
    """Log token usage per call so quota consumption stays visible."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    logger.debug(
        "[usage] prompt=%s output=%s total=%s",
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
    )


def ask(question: str) -> str:
    response = generate(model=MODEL, contents=question)
    return response.text


if __name__ == "__main__":
    print(ask("Say hello in one short sentence."))