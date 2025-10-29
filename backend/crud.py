from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import Session
from datetime import datetime
from database import Base
from schemas import ImageResult
import torch

# DB Model
class ImageResultModel(Base):
    __tablename__ = "image_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

# CRUD
def create_image_result(db: Session, filename: str, label: str, confidence: float):
    # tensor to float
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.item()

    db_item = ImageResultModel(
        filename=filename,
        label=label,
        confidence=confidence
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_image_results(db: Session, skip: int = 0, limit: int = 100):
    return db.query(ImageResultModel).offset(skip).limit(limit).all()