from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, LargeBinary, ForeignKey, event
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import Engine
import os

# --- Constants ---
METADATA_DB_NAME = "metadata.db"
THUMBNAILS_DB_NAME = "thumbnails.db"

# --- Database Setup ---
BaseMetadata = declarative_base()
BaseThumbnails = declarative_base()

# --- Enable WAL Mode for Concurrency ---
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

# --- Schema: Metadata DB ---
class Image(BaseMetadata):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    original_path = Column(String, nullable=True)  # Where the file was FIRST found (never changes)
    processed = Column(Integer, default=0) # 0=Pending, 1=Done, -1=Error
    # Using SHA-256 hash if needed later, but keeping it simple for now
    
    faces = relationship("Face", back_populates="image", cascade="all, delete-orphan")

class Face(BaseMetadata):
    __tablename__ = 'faces'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    embedding = Column(LargeBinary, nullable=False) # Serialized numpy array
    bbox = Column(String, nullable=False) # JSON string of [x, y, w, h]
    det_score = Column(Float, nullable=False)
    
    # Clustering fields
    cluster_id = Column(Integer, nullable=True, index=True)
    is_verified = Column(Boolean, default=False)
    
    image = relationship("Image", back_populates="faces")

class Person(BaseMetadata):
    __tablename__ = 'people'
    
    cluster_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    
    # Incremental Learning
    mean_embedding = Column(LargeBinary, nullable=True) # The 512-D centroid (numpy array)
    face_count = Column(Integer, default=0) # How many faces went into this mean

# --- Schema: Thumbnails DB ---
class FaceThumbnail(BaseThumbnails):
    __tablename__ = 'face_thumbnails'
    
    id = Column(Integer, primary_key=True)
    face_id = Column(Integer, unique=True, nullable=False) # Linked to Face.id manually
    thumbnail = Column(LargeBinary, nullable=False) # JPEG bytes

# --- Initialization Utilities ---
def get_db_engines():
    # Echo=False to avoid spamming the console
    engine_meta = create_engine(f'sqlite:///{METADATA_DB_NAME}', echo=False)
    engine_thumb = create_engine(f'sqlite:///{THUMBNAILS_DB_NAME}', echo=False)
    
    return engine_meta, engine_thumb

def init_dbs():
    engine_meta, engine_thumb = get_db_engines()
    
    # Create tables
    BaseMetadata.metadata.create_all(engine_meta)
    BaseThumbnails.metadata.create_all(engine_thumb)
    
    return sessionmaker(bind=engine_meta), sessionmaker(bind=engine_thumb)
