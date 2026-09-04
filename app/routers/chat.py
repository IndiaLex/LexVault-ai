from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import answer_query

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    user_roles: list[str] = ["default"]

@router.post("/ai/chat")
def chat(req: ChatRequest):
    return answer_query(req.query, req.user_roles)