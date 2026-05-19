# Face Clustering Walkthrough: A Beginner's Guide

Welcome to the Local AI Face Sorter! If you are looking at this repository for the first time, all the different scripts and databases can look a bit overwhelming. 

This guide will walk you through exactly how data flows through the system, step-by-step. 

---

## Phase 1: Reading Your Photos (The Indexer)

Everything starts with **`indexer.py`**. You point this script at a folder on your hard drive full of thousands of photos.

1. **Scanning**: The script scans the folder and finds every image file. *(Tip: If you are looking for a great dataset to test this software on, we highly recommend the [105 Classes Pins Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) on Kaggle!)*
2. **AI Detection**: It passes each image to the `InsightFace` AI model running on your local machine.
3. **The Fingerprint**: For every face it finds, the AI generates a **512-dimensional vector** (called an embedding). You can think of this as a unique mathematical fingerprint for that specific face.
4. **Saving**: 
   - It saves the mathematical fingerprint and the original file path into `metadata.db`.
   - It crops out just the face into a tiny JPEG thumbnail and saves it into `thumbnails.db` (so the web UI can load faces instantly later!).

> **⚡ Hardware Acceleration Note**: The `indexer.py` script is smart! It automatically scans your computer to find the fastest hardware available. If you are on a Mac, it will use your Apple Silicon Neural Engine (CoreML). If you are on a Windows PC with an NVIDIA GPU, make sure you run `pip install onnxruntime-gpu` so the script can detect and use CUDA to process thousands of faces per minute! If you just have a standard CPU, it will gracefully fall back to CPU processing without crashing.

---

## Phase 2: Sorting and Viewing (The Server & UI)

Once your photos are indexed, you start the web dashboard by running **`server.py`**. 

1. **The Web Interface**: `server.py` hosts a local Flask website that you can open in your browser. 
2. **Clustering (`ml_core.py`)**: When you click the **"Re-Cluster Photos"** button in the UI, `server.py` asks `ml_core.py` to do the heavy lifting.
   - `ml_core.py` grabs all the 512D math vectors from the database.
   - It runs an algorithm called **DBSCAN** to group faces that are mathematically close together into "Clusters" (groups of the same person).
3. **Display**: The server then sends these clusters to your web browser, where you can see groups of people, name them ("Mom", "Dad"), or delete bad photos.

---

## Phase 3: The Race Detection Neural Network

We didn't just want to sort faces; we wanted the AI to understand demographics! Here is how the race detection pipeline was built:

1. **Extracting Training Data (`extract_fairface.py`)**: 
   We downloaded the [FairFace Dataset](https://github.com/dchen236/FairFace) (a massive public dataset of ~100,000 diverse faces balanced across 7 demographic groups). We wrote this script to pass every single one of those 100,000 faces through InsightFace to get their 512D fingerprints, and saved them alongside their true race labels (0-6).
2. **Training the Brain (`train_race.py`)**: 
   We built a lightweight TensorFlow Neural Network. We fed it the 100,000 fingerprints we extracted. It learned how to mathematically map a 512D face vector to one of 7 races. The resulting "brain" was saved as a tiny 2MB file: `race_classifier.keras`.
3. **Production Inference (`ml_core.py` & `server.py`)**: 
   - Now, when you open the web UI, `server.py` looks at the clusters you generated in Phase 2.
   - For every cluster, it sends *all* the faces in that cluster to `ml_core.predict_race()`.
   - `ml_core` loads the `race_classifier.keras` brain and predicts the race for every single photo in milliseconds. 
   - It uses a **Majority Vote** (if 9 out of 10 photos of a person predict "White", the cluster is labeled "White"). This perfectly cancels out blurry photos or bad angles!
   - `server.py` attaches this string to the cluster, and the UI displays it as a sleek badge next to the person's name! (You can also right-click any individual photo to get a standalone AI prediction just for that image).

---

## Phase 4: Utility Scripts

There are a few other helper scripts in the repository that you will use once you finish sorting:

* **`exporter.py`**: Once you are happy with how everyone is named on the website, you run this script. It reads the database and automatically creates actual folders on your hard drive (e.g., `Output/Mom/`, `Output/Dad/`) and links your original photos into them! If a photo has both Mom and Dad in it, it seamlessly puts the photo in both folders without taking up extra disk space.
* **`reset_clusters.py`**: If you ever make a mistake or want to start naming people from scratch, you run this script. It safely deletes all your names and groups from the database, *without* forcing you to wait for `indexer.py` to extract the fingerprints all over again.
* **`database.py`**: This isn't a script you run directly; it just defines the SQLAlchemy tables (like `Face`, `Person`, and `Image`) so Python knows how to talk to the `.db` files.
