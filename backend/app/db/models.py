from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    filename        = Column(String(255), nullable=True)
    predicted_class = Column(String(50), nullable=False)
    predicted_index = Column(Integer, nullable=False)
    confidence      = Column(Float, nullable=False)
    prob_normal     = Column(Float, nullable=False)
    prob_pneumonia  = Column(Float, nullable=False)
    model_name      = Column(String(100), nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
