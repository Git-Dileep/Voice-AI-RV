from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from brain.orchestrator import AgentOrchestrator

app = FastAPI(title="Voice AI Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AgentOrchestrator()


class UserInput(BaseModel):
    text: str
    lang: str = "en"
    confidence: float = 1.0


@app.post("/api/chat")
def chat(request: UserInput):
    response = orchestrator.process_input(
        request.text,
        request.lang,
        request.confidence
    )

    return {
        "ok": True,
        "response": response
    }


@app.get("/")
def root():
    return {"message": "Orchestrator API is running"}