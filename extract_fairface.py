import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from rich.progress import track
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

def extract_features(csv_path, img_base_dir, output_prefix):
    """
    Loops through the FairFace dataset, extracts the 512D embeddings, and saves them.
    Features auto-save and resume capabilities.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please download the dataset first!")
        return

    X_file = f"{output_prefix}_X.npy"
    y_file = f"{output_prefix}_y.npy"
    idx_file = f"{output_prefix}_idx.txt"
    
    start_idx = 0
    if os.path.exists(X_file) and os.path.exists(y_file) and os.path.exists(idx_file):
        print(f"\n[RESUME] Found existing data for {output_prefix}. Resuming from checkpoint...")
        X = np.load(X_file).tolist()
        y = np.load(y_file).tolist()
        with open(idx_file, "r") as f:
            start_idx = int(f.read().strip())
    else:
        X = []
        y = []

    print("Loading AI Models...")
    # Auto-detect optimal ONNX hardware acceleration
    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available: providers.append('CUDAExecutionProvider')
    elif 'CoreMLExecutionProvider' in available: providers.append('CoreMLExecutionProvider')
    elif 'DmlExecutionProvider' in available: providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print(f"Reading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if start_idx >= len(df):
        print(f"Already fully processed {output_prefix}!")
        return
        
    print(f"Starting extraction from image {start_idx} out of {len(df)}...")
    
    df_remaining = df.iloc[start_idx:]
    actual_idx = start_idx
    
    try:
        for row in track(df_remaining.itertuples(), total=len(df_remaining), description=f"Processing {output_prefix}"):
            img_path = os.path.join(img_base_dir, row.file)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Pad the image just like we did in indexer.py
                    padded_img = pad_image(img)
                    faces = app.get(padded_img)
                    
                    # Only process if it successfully found exactly 1 face
                    if len(faces) == 1:
                        X.append(faces[0].embedding.astype(np.float32))
                        y.append(RACE_MAP[row.race])
            
            actual_idx += 1
            
            # Auto-save every 1000 images in case of a crash
            if actual_idx % 1000 == 0:
                np.save(X_file, np.array(X))
                np.save(y_file, np.array(y))
                with open(idx_file, "w") as f:
                    f.write(str(actual_idx))
                    
    except KeyboardInterrupt:
        print(f"\n\n[PAUSED] Process stopped by user at image {actual_idx}!")
        print("Saving progress so you can resume later...")
        
    # Save final state before exiting
    np.save(X_file, np.array(X))
    np.save(y_file, np.array(y))
    with open(idx_file, "w") as f:
        f.write(str(actual_idx))
        
    print(f"Successfully saved {len(X)} faces to {X_file} and {y_file}")

if __name__ == "__main__":
    # NOTE: Change these paths based on where you unzip the FairFace dataset!
    TRAIN_CSV = "fairface_label_train.csv"
    VAL_CSV = "fairface_label_val.csv"
    DATASET_DIR = "./fairface-img-margin025-trainval/" 
    
    # Process the ENTIRE dataset
    print("=== Extracting Training Data ===")
    extract_features(TRAIN_CSV, DATASET_DIR, output_prefix="fairface_train")
    
    print("\n=== Extracting Validation Data ===")
    extract_features(VAL_CSV, DATASET_DIR, output_prefix="fairface_val")
