from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(".env", override=True)
key = os.getenv("XAI_API_KEY", "").strip()

print("Key loaded:", bool(key))
print("Key length:", len(key))

if not key:
    raise SystemExit("XAI_API_KEY was not found in backend/.env")

try:
    client = OpenAI(
        api_key=key,
        base_url="https://api.x.ai/v1"
    )
    client.models.list()
    print("SUCCESS: xAI accepted the API key.")
except Exception as e:
    print("API test failed:", type(e).__name__, str(e))
