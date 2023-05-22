from pydantic import BaseModel

class Message(BaseModel):
    message: str
    id: int | None = None
    insertDate: str
    updateDate: str | None = None
    spamOrNot: str | None = None
