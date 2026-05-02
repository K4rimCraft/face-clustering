import os
import argparse
import sys
import numpy as np
import cv2
from sqlalchemy.orm import Session
from sqlalchemy import select
from insightface.app import FaceAnalysis

# Rich for UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# Local imports
from database import init_dbs, Image, Face, FaceThumbnail

# --- Configuration ---
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MIN_DET_SCORE = 0.60
THUMBNAIL_SIZE = (128, 128)
BATCH_COMMIT_SIZE = 50

# Global Console
console = Console()

def get_image_files(root_dir):
    """Recursively yield absolute paths to image files."""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                yield os.path.join(root, file)

def generate_thumbnail(img, bbox):
    """
    Crops the face with some padding and resizes to 128x128.
    Returns JPEG bytes.
    """
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add 20% padding
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    face_crop = img[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return None
        
    # Resize
    try:
        face_crop = cv2.resize(face_crop, THUMBNAIL_SIZE)
        _, encoded = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return encoded.tobytes()
    except Exception:
        return None

def process_image(app, file_path, session_meta, session_thumb):
    """Reads image, detects faces, saves to DB."""
    try:
        # Load image with OpenCV
        # Handling non-ascii paths by reading as bytes then decoding is sometimes needed,
        # but standard cv2.imread works on modern Python 3.10+ for Windows mostly.
        # If path issues arise, we can switch to numpy fromfile.
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return False # Corrupt or unreadable
            
        faces = app.get(img)
        
        # 1. Create Image Record (original_path = path when first discovered)
        db_image = Image(path=file_path, original_path=file_path, processed=1)
        session_meta.add(db_image)
        session_meta.flush() # Flush to get db_image.id
        
        face_count = 0
        
        for face in faces:
            if face.det_score < MIN_DET_SCORE:
                continue
                
            # 2. Serialize Embedding
            embedding_bytes = face.embedding.tobytes()
            bbox = face.bbox
            
            # 3. Create Face Record
            db_face = Face(
                image_id=db_image.id,
                embedding=embedding_bytes,
                bbox=str(list(bbox)), # Store as string representation of list
                det_score=float(face.det_score)
            )
            session_meta.add(db_face)
            session_meta.flush() # Flush to get db_face.id
            
            # 4. Create Thumbnail
            thumb_bytes = generate_thumbnail(img, bbox)
            if thumb_bytes:
                db_thumb = FaceThumbnail(
                    face_id=db_face.id,
                    thumbnail=thumb_bytes
                )
                session_thumb.add(db_thumb)
            
            face_count += 1
            
        return True
        
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")
        return False

def main():
    parser = argparse.ArgumentParser(description="Face Indexer for Local Sorting Pipeline")
    parser.add_argument("input_dir", help="Path to the directory containing photos")
    parser.add_argument("--reset", action="store_true", help="Clear databases before starting")
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    
    if not os.path.exists(input_dir):
        console.print(f"[bold red]Directory not found: {input_dir}[/bold red]")
        sys.exit(1)
        
    # --- DB Init ---
    if args.reset:
        if os.path.exists("metadata.db"): os.remove("metadata.db")
        if os.path.exists("thumbnails.db"): os.remove("thumbnails.db")
        console.print("[yellow]Databases reset.[/yellow]")

    SessionMeta, SessionThumb = init_dbs()
    
    # --- InsightFace Init ---
    console.print("[cyan]Initializing InsightFace (DirectML)...[/cyan]")
    # Using specific options to ensure DML usage
    app = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    console.print("[green]AI Ready![/green]")
    
    # --- Discovery Phase ---
    session_meta = SessionMeta()
    
    # Get set of already processed paths
    console.print("[cyan]Scanning for existing records...[/cyan]")
    existing_paths = set(
        path for (path,) in session_meta.query(Image.path).all()
    )
    
    console.print(f"Found {len(existing_paths)} already indexed images.")
    
    # Discovery
    console.print(f"[cyan]Scanning directory: {input_dir}[/cyan]")
    files_to_process = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Finding files...", total=None)
        
        for file_path in get_image_files(input_dir):
            if file_path not in existing_paths:
                files_to_process.append(file_path)
    
    console.print(f"[bold green]Found {len(files_to_process)} new files to process.[/bold green]")
    
    if not files_to_process:
        console.print("Nothing to do!")
        sys.exit(0)
        
    # --- Processing Loop ---
    session_thumb = SessionThumb()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Indexing...", total=len(files_to_process), filename="Starting...")
        
        count = 0
        for i, file_path in enumerate(files_to_process):
            filename = os.path.basename(file_path)
            progress.update(task, filename=filename)
            
            success = process_image(app, file_path, session_meta, session_thumb)
            
            # Commit periodically
            if (i + 1) % BATCH_COMMIT_SIZE == 0:
                session_meta.commit()
                session_thumb.commit()
                
            progress.advance(task)
            
        # Final commit
        session_meta.commit()
        session_thumb.commit()

    console.print("[bold green]Indexing Complete![/bold green]")

if __name__ == "__main__":
    main()
