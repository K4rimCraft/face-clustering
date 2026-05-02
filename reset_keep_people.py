"""
Reset Script - Keep People, Clear Everything Else
==================================================
Clears images, faces, and thumbnails while preserving
the People table (names and learned face embeddings).

This allows you to re-scan with fresh photos while
keeping your name suggestions working.
"""

import os
from sqlalchemy import create_engine, text
from database import METADATA_DB_NAME, THUMBNAILS_DB_NAME

def reset_keep_people():
    print("=" * 50)
    print("RESET: Keep People, Clear Faces/Images")
    print("=" * 50)
    
    # Show what will be preserved
    engine = create_engine(f'sqlite:///{METADATA_DB_NAME}')
    with engine.connect() as conn:
        people_count = conn.execute(text("SELECT COUNT(*) FROM people")).scalar()
        images_count = conn.execute(text("SELECT COUNT(*) FROM images")).scalar()
        faces_count = conn.execute(text("SELECT COUNT(*) FROM faces")).scalar()
    
    print(f"\nCurrent state:")
    print(f"  - People: {people_count} (WILL BE KEPT)")
    print(f"  - Images: {images_count} (will be deleted)")
    print(f"  - Faces:  {faces_count} (will be deleted)")
    
    # Confirm
    response = input("\nProceed with reset? (type 'yes' to confirm): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    print("\nClearing faces table...")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM faces"))
        conn.commit()
    
    print("Clearing images table...")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM images"))
        conn.commit()
    
    print("Clearing thumbnails database...")
    if os.path.exists(THUMBNAILS_DB_NAME):
        engine_thumb = create_engine(f'sqlite:///{THUMBNAILS_DB_NAME}')
        with engine_thumb.connect() as conn:
            conn.execute(text("DELETE FROM face_thumbnails"))
            conn.commit()
    
    # Verify
    with engine.connect() as conn:
        people_count = conn.execute(text("SELECT COUNT(*) FROM people")).scalar()
        images_count = conn.execute(text("SELECT COUNT(*) FROM images")).scalar()
        faces_count = conn.execute(text("SELECT COUNT(*) FROM faces")).scalar()
    
    print("\n" + "=" * 50)
    print("RESET COMPLETE!")
    print("=" * 50)
    print(f"\nNew state:")
    print(f"  - People: {people_count} (preserved!)")
    print(f"  - Images: {images_count}")
    print(f"  - Faces:  {faces_count}")
    print(f"\nYou can now run: python indexer.py <your_folder>")
    print("New faces will get suggestions based on your saved people!")


if __name__ == "__main__":
    reset_keep_people()
