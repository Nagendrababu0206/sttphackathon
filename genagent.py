from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

load_dotenv()

class Question(BaseModel):
    Question_number: int
    Question_body: str
    Question_options: str
    Correct_ans: int

class Question_list(BaseModel):
    All_questions: List[Question]

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

res = client.models.generate_content(
    model="gemma-4-31b-it",
    contents="Give me 5 questions on Generative ai",
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Question_list
    )
)

print(res.text)
