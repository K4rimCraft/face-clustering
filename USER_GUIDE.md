# User Guide: From Scattered to Sorted

This guide walks you through the exact steps to organize your photo collection. Follow these phases in order.

---

## Phase 1: Preparation
*Do this once.*

1.  **Open your Terminal** (Command Prompt or PowerShell).
2.  **Navigate to your project folder**:
    ```powershell
    cd i:\scrapers\newerpicsorter
    ```
3.  **Activate the Environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
    *(You should see `(venv)` at the start of your line).*
4.  **Verify your System**:
    ```powershell
    python check_gpu_env.py
    ```
    *It should say "Success! DirectML is available".*

---

## Phase 2: Indexing (The Harvest)
*Goal: Find every photo you own and let the AI see it.*

1.  **Run the Indexer** on your first location:
    ```powershell
    python indexer.py "D:\Old_Hard_Drive\Photos"
    ```
    *   *What happens?* It scans files, detects faces, and saves them to the database.
    *   *Time:* ~10-50 photos per second. You can stop it (`Ctrl+C`) and resume later; it skips files it already knows.

2.  **Repeat** for other locations:
    ```powershell
    python indexer.py "C:\Users\Kimos\Downloads"
    python indexer.py "F:\Backup2015"
    ```

---

## Phase 3: The Brain (Sorting & Naming)
*Goal: Tell the AI who "Mom" and "Dad" are.*

1.  **Start the Web Server**:
    ```powershell
    python server.py
    ```
    *Keep this terminal window open.*

2.  **Open your Browser**:
    Go to: [http://localhost:5000](http://localhost:5000)

3.  **First Run**:
    *   You will probably see "0 faces" or empty clusters.
    *   Click the **"Re-Cluster Photos"** button in the sidebar.
    *   *Wait...* (Watch the terminal for progress).

4.  **Naming Strategy**:
    *   Scroll through the clusters.
    *   **Cluster 1** has 500 photos of a woman? Type **"Mom"** in the name box.
    *   **Cluster 2** has 300 photos of a man? Type **"Dad"**.
    *   **Cluster 15** has 5 photos of Mom (maybe wearing sunglasses)? Type **"Mom"** again.
        *   *Magic:* Naming it "Mom" merges it with Cluster 1 and updates the AI's idea of what Mom looks like.

5.  **Refining**:
    *   Use the **EPS Slider**.
        *   Faces too mixed up? **Decrease EPS** (e.g., 0.40). Click Re-Cluster.
        *   Mom split into 20 tiny groups? **Increase EPS** (e.g., 0.50). Click Re-Cluster.
    *   **Remove Bad Faces**: Hover over a face and click the red **X** if it's not them.

---

## Phase 4: The Clean Up (Exporting)
*Goal: Move files from their scattered mess into clean folders.*

1.  **Stop the Server**:
    Go to your terminal and press `Ctrl+C`.

2.  **Run the Exporter**:
    Decide where you want your final library to be (e.g., `D:\My_Life_Sorted`).

    **Option A: The Safe Copy (Recommended)**
    *   Copies files. Originals stay untouched. Requires more disk space.
    ```powershell
    python exporter.py "D:\My_Life_Sorted" --mode copy
    ```

    **Option B: The Space Saver (Hard Links)**
    *   Files appear in new folder but take **0 space**. Originals must stay where they are.
    ```powershell
    python exporter.py "D:\My_Life_Sorted" --mode link
    ```

    **Option C: The "Move Everything" (Destructive)**
    *   Moves files out of scattered folders into the new one. Updates the Database so the App still works.
    ```powershell
    python exporter.py "D:\My_Life_Sorted" --mode move
    ```

3.  **Check your Output Folder**:
    Open `D:\My_Life_Sorted`. You will see folders: `Mom`, `Dad`, etc.

---

## Phase 5: Future Usage
*Goal: Adding yesterday's party photos.*

1.  **Index the new folder**:
    ```powershell
    python indexer.py "E:\SD_Card\Party"
    ```
2.  **Run Server**: `python server.py`
3.  **Click Re-Cluster**:
    *   Since you already named "Mom" and "Dad", the AI will likely find them in the new photos and **automatically suggest** their names.
    *   Confirm the names.
4.  **Run Exporter**:
    ```powershell
    python exporter.py "D:\My_Life_Sorted" --mode move
    ```
    *   Only the *new* files are moved.

---

## Emergency / Troubleshooting

*   **Database Error?**
    Run: `python migrate_db.py` to fix schema issues.
*   **"Failed to Fetch"?**
    Restart `python server.py`.
*   **Want to start over completely?**
    Delete `metadata.db` and `thumbnails.db`. (Original photos are safe).
