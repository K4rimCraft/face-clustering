import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import warnings

# Hide warnings from insightface
warnings.filterwarnings("ignore", category=FutureWarning)

# FairFace has 7 races, we need to map them to numbers (0-6) for the neural network
RACE_MAP = {
    'White': 0,
    'Black': 1,
    'Indian': 2,
    'East Asian': 3,
    'Southeast Asian': 4,
    'Middle Eastern': 5,
    'Latino_Hispanic': 6
}

def pad_image(img, pad_percent=0.5):
    """Pads the image with a black border to help InsightFace detect tightly cropped faces."""
    pad_h = int(img.shape[0] * pad_percent)
    pad_w = int(img.shape[1] * pad_percent)
    return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

def extract_features(csv_path, img_base_dir, output_prefix, limit=None):
    """
    Loops through the FairFace dataset, extracts the 512D embeddings, and saves them.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please download the dataset first!")
        return

    print("Loading AI Models...")
    # Load InsightFace (Using CoreML for your Mac!)
    app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print(f"Reading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # For testing, you might not want to process all 86,000 images right away
    if limit:
        print(f"Limiting to first {limit} images for testing...")
        df = df.head(limit)
        
    X = [] # This will hold the 512-dimension arrays
    y = [] # This will hold the race IDs (0-6)
    
    print("Starting extraction...")
    
    for idx, row in enumerate(df.itertuples()):
        # The 'file' column in FairFace looks like "train/1.jpg"
        img_path = os.path.join(img_base_dir, row.file)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Pad the image just like we did in indexer.py
        padded_img = pad_image(img)
        
        # Scan for faces
        faces = app.get(padded_img)
        
        # Only process if it successfully found exactly 1 face
        if len(faces) == 1:
            X.append(faces[0].embedding.astype(np.float32))
            y.append(RACE_MAP[row.race])
            
        # Print a status update every 1000 images
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} / {len(df)} images...")
            
    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nExtraction complete! Successfully processed {len(X)} faces.")
    
    # Save the data to disk
    np.save(f"{output_prefix}_X.npy", X)
    np.save(f"{output_prefix}_y.npy", y)
    print(f"Saved data to {output_prefix}_X.npy and {output_prefix}_y.npy")

if __name__ == "__main__":
    # NOTE: Change these paths based on where you unzip the FairFace dataset!
    TRAIN_CSV = "fairface_label_train.csv"
    VAL_CSV = "fairface_label_val.csv"
    DATASET_DIR = "." 
    
    # Change limit=None to process the ENTIRE dataset
    # We set it to 5000 right now so you can run a quick 2-minute test.
    print("=== Extracting Training Data ===")
    extract_features(TRAIN_CSV, DATASET_DIR, output_prefix="fairface_train", limit=5000)
    
    print("\n=== Extracting Validation Data ===")
    extract_features(VAL_CSV, DATASET_DIR, output_prefix="fairface_val", limit=1000)
