from pydantic import BaseModel
from datetime import datetime

# Upload Schema
class ImageUpload(BaseModel):
    filename: str

# DB Save Schema
class ImageResult(BaseModel):
    id: int
    filename: str
    label: str
    confidence: float
    uploaded_at: datetime

    class Config:
        orm_mode = True