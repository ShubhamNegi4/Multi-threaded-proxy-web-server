from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json

DATABASE_URL = "sqlite:///./url_logs.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class URLLog(Base):
    __tablename__ = "url_logs"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    reasons = Column(Text)  # Store as JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
     def get_reasons(self):
        """Get reasons as list"""
        if self.reasons:
            return json.loads(self.reasons)
        return []

    def set_reasons(self, reasons_list):
        """Set reasons as JSON string"""
        self.reasons = json.dumps(reasons_list)
    
   

# Create table if it doesn't exist
Base.metadata.create_all(bind=engine)
