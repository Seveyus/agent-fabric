from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.config import settings

engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
