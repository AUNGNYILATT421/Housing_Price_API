import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, JSON, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

# SQLite needs a special connect arg for multithreading
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    input_features = Column(JSON, nullable=False)
    predicted_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, input_features: dict, predicted_price: float) -> Prediction:
    record = Prediction(input_features=input_features, predicted_price=predicted_price)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_predictions(db: Session, limit: int = 50) -> list[Prediction]:
    return db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
