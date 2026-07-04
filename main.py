import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
my_client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
res=my_client.models.generate_content(model="gemini-2.5-flash-lite",contents="how are you")
print(res.text)
