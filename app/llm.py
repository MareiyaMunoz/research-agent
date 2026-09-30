import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env into environment variables

client = genai.Client()  # automatically finds GEMINI_API_KEY
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def ask(question: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
    )
    return response.text

if __name__ == "__main__":
    print(ask("Say hello in one short sentence."))