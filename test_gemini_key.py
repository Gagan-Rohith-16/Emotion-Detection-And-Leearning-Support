from dotenv import load_dotenv
import os

load_dotenv()

print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))