import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Tên model: {m.name}")
except Exception as e:
    print(f"Lỗi khi lấy danh sách: {e}")