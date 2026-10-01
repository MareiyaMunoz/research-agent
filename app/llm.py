import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()  # reads .env into environment variables

client = genai.Client()  # automatically finds GEMINI_API_KEY
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

RETRYABLE = {429, 500, 503, 504}  # temporary errors worth retrying
MAX_ATTEMPTS = 5


def generate(**kwargs):
    """Call Gemini, retrying temporary errors with increasing waits."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.models.generate_content(**kwargs)
        except errors.APIError as e:
            is_last = attempt == MAX_ATTEMPTS - 1
            if e.code not in RETRYABLE or is_last:
                raise  # not temporary, or out of attempts
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"Gemini error {e.code}, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...")
            time.sleep(wait)


def ask(question: str) -> str:
    response = generate(model=MODEL, contents=question)
    return response.text


if __name__ == "__main__":
    print(ask("Say hello in one short sentence."))