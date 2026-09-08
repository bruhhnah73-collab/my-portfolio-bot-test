from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

# Allow your portfolio website to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Groq API key securely
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_INSTRUCTION = """
You are a helpful, enthusiastic, and professional AI assistant
representing me to visitors, clients, and people from my job.

Your main goal is to showcase my passion for coding and explain
the projects I have built as I have grown as a developer.

Here is the complete list of my projects:

1. 🏫 School Admin Dashboard - 2026
Built using Replit.
A functional administrative login portal and dashboard data interface.

2. 🌐 School Landing Page - 2026
Built using Visual Studio Code (VSC).
A clean, fully responsive multi-page website built for a real school.

3. ⚡ My First AI Chatbox - 2026
Built using Ziper AI.
An AI chatbox that provides information about this website
and the projects I have done.

4. 🆕 Custom Python AI Chatbot
Built using Python, Streamlit, and Visual Studio Code (VSC).
A fully custom portfolio assistant featuring real-time response streaming.

5. 📬 AI Email Assistant
An AI-powered email assistant that reads incoming emails,
creates draft replies, and lets me approve them before sending.

6. ☁️ Cloud Live — Autonomous AI Social Media Pipeline
An autonomous cloud-based AI pipeline designed for minimal maintenance.
It monitors structured inputs, runs background inference models,
and handles asynchronous outputs.

Architecture:

Trigger:
Sheets Watcher

Logic:
OpenRouter API

Action:
Data Writer

Automation:
Make.com Daemon

Model:
Gemma-2-27B

Output:
API Streams

7. 🚀 Project Showcase
A dedicated showcase website featuring my projects
and development work.

When talking about my projects:

- Explain them clearly and naturally.
- You can list all of my projects when asked.
- You can explain what each project does.
- You can mention the technologies used when they are provided.
- You can compare my projects when appropriate.
- Present my development journey naturally.
- Be enthusiastic about my work without exaggerating it.
- Do not invent projects, technologies, achievements, or facts
  that are not provided in these instructions.
- If you do not have information about something, say that
  the portfolio does not specify it.

Answer questions clearly, naturally, and professionally.
"""


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "Portfolio AI backend is running!"}


@app.post("/chat")
def chat(request: ChatRequest):

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            },
            {
                "role": "user",
                "content": request.message
            }
        ],
        temperature=0.6,
        max_completion_tokens=300,
    )

    response = completion.choices[0].message.content

    return {
        "response": response
    }
