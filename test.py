# test_models.py
import google.generativeai as genai
import os

# Make sure your GEMINI_API_KEY is set
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()
for m in models:
    print(m.name, getattr(m, "description", ""))
