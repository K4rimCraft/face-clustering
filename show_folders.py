"""
Show Scanned Folders
====================
Lists all unique folders that have been indexed, with image counts.
"""
import os
import sqlite3
from collections import defaultdict

DB_PATH = "metadata.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all image paths
    cursor.execute("SELECT path FROM images")
    rows = cursor.fetchall()
    
    if not rows:
        print("No images in database.")
        conn.close()
        return
    
    # Count images per folder
    folder_counts = defaultdict(int)
    for (path,) in rows:
        folder = os.path.dirname(path)
        folder_counts[folder] += 1
    
    # Sort by count (descending)
    sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"  SCANNED FOLDERS ({len(sorted_folders)} unique folders)")
    print(f"{'='*60}\n")
    
    total_images = 0
    for folder, count in sorted_folders:
        print(f"  {count:>6} images  │  {folder}")
        total_images += count
    
    print(f"\n{'─'*60}")
    print(f"  TOTAL: {total_images} images across {len(sorted_folders)} folders")
    print(f"{'─'*60}\n")
    
    conn.close()

if __name__ == "__main__":
    main()
