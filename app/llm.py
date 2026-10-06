import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

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


def generate(**kwargs):
    """Call Gemini, retrying temporary errors with increasing waits."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(**kwargs)
            _log_usage(response)
            return response
        except errors.APIError as e:
            is_last = attempt == MAX_ATTEMPTS - 1
            if e.code not in RETRYABLE or is_last:
                raise  # not temporary, or out of attempts
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"Gemini error {e.code}, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...")
            time.sleep(wait)


def _log_usage(response) -> None:
    """Print token usage so quota consumption is visible per call."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    print(
        f"[usage] prompt={usage.prompt_token_count} "
        f"output={usage.candidates_token_count} "
        f"total={usage.total_token_count}"
    )


def ask(question: str) -> str:
    response = generate(model=MODEL, contents=question)
    return response.text


if __name__ == "__main__":
    print(ask("Say hello in one short sentence."))