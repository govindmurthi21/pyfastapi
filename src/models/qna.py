from pydantic import BaseModel

class QnA(BaseModel):
    question: str
    id: int | None = None
    insertDate: str
    updateDate: str | None = None
    answer: str | None = None
    sqsMessageId: str | None = None