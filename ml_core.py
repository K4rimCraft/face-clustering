import numpy as np
import os
import tensorflow as tf
from collections import deque

# ==========================================
# RACE CLASSIFIER (TensorFlow)
# ==========================================
RACE_LABELS = ['White', 'Black', 'Indian', 'East Asian', 'Southeast Asian', 'Middle Eastern', 'Latino/Hispanic']

# Load the trained Keras model once when the module starts
_race_model = None
model_path = "race_classifier.keras"
if os.path.exists(model_path):
    _race_model = tf.keras.models.load_model(model_path)

def predict_race(embeddings):
    """
    Predicts the race of a cluster of 512D face embeddings using a majority vote.
    """
    if _race_model is None or not embeddings:
        return "Unknown"
        
    # Ensure inputs are a 2D array (batch)
    x = np.array(embeddings)
    if x.ndim == 1:
        x = x.reshape(1, -1)
        
    # Run the model on the entire batch at once (super fast!)
    probabilities = _race_model.predict(x, verbose=0)
    
    # Get the predicted index for each face in the cluster
    best_indices = np.argmax(probabilities, axis=1)
    
    # Find the most common prediction (Majority Vote)
    values, counts = np.unique(best_indices, return_counts=True)
    mode_index = values[np.argmax(counts)]
    
    return RACE_LABELS[mode_index]

# ==========================================
# MATH & CLUSTERING LOGIC
# ==========================================


def calculate_centroid(embeddings):
    """
    Calculates the mean embedding for a list of face embeddings.
    """
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)

def calculate_weighted_centroid(mean1, count1, mean2, count2):
    """
    Calculates the weighted average of two centroids.
    Used when merging clusters or renaming.
    """
    return (mean1 * count1 + mean2 * count2) / (count1 + count2)

def calculate_confidence(dist, min_dist=0.4, max_dist=1.0):
    """
    Converts a cosine distance to a confidence percentage (0-100).
    """
    dist = float(dist)
    if dist < min_dist:
        return 100.0
    elif dist > max_dist:
        return 0.0
    else:
        return (1.0 - dist) / (1.0 - min_dist) * 100

def custom_cosine_distances(A, B):
    """
    Computes pairwise cosine distances between two matrices A and B from scratch.
    """
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    
    norm_A = np.maximum(norm_A, 1e-10)
    norm_B = np.maximum(norm_B, 1e-10)
    
    normalized_A = A / norm_A
    normalized_B = B / norm_B
    
    similarities = np.dot(normalized_A, normalized_B.T)
    distances = 1.0 - similarities
    return np.clip(distances, 0.0, 2.0)

def cluster_faces(embeddings, eps=0.45, min_samples=3):
    """
    A custom implementation of the DBSCAN clustering algorithm.
    Groups facial embeddings based on cosine distance density.
    """
    if not len(embeddings):
        return []
        
    X = np.array(embeddings)
    n_samples = X.shape[0]
    
    # Calculate the full distance matrix for all points
    distances = custom_cosine_distances(X, X)
    
    # Precompute the epsilon neighborhood for every point upfront
    # so we don't have to recalculate distances during the loop
    neighbors_list = [np.where(distances[i] <= eps)[0] for i in range(n_samples)]
    
    labels = np.full(n_samples, -2) # -2 indicates unvisited
    cluster_id = 0
    
    for i in range(n_samples):
        if labels[i] != -2:
            continue
            
        # Retrieve the precomputed neighbors for the current point
        neighbors = neighbors_list[i] 
        
        if len(neighbors) < min_samples:
            labels[i] = -1 # Mark as Noise
            continue
            
        # Core point found, start a new cluster
        labels[i] = cluster_id
        
        # Initialize a double-ended queue (deque) with the neighbors 
        # to efficiently expand the cluster outward
        queue = deque([n for n in neighbors if n != i])
        in_queue = np.zeros(n_samples, dtype=bool)
        for n in queue:
            in_queue[n] = True
            
        while queue:
            # Process the next point in the queue
            p = queue.popleft() 
            in_queue[p] = False
            
            if labels[p] == -1:
                # Was previously marked as noise, now a border point
                labels[p] = cluster_id
                
            elif labels[p] == -2:
                labels[p] = cluster_id
                
                # Check if this border point is also a core point
                p_neighbors = neighbors_list[p]
                
                if len(p_neighbors) >= min_samples:
                    # Add its unvisited neighbors to the queue to keep expanding
                    for n in p_neighbors:
                        if (labels[n] == -2 or labels[n] == -1) and not in_queue[n]:
                            queue.append(n)
                            in_queue[n] = True
                            
        cluster_id += 1
        
    # Mark any remaining unvisited points as noise
    labels[labels == -2] = -1
    return labels

def get_top_matches(target_embedding, known_centroids, top_k=5):
    """
    Compares a target embedding against known centroids and returns the top matches.
    Returns: list of dicts with 'cluster_id', 'name', 'confidence', 'distance'.
    """
    if not known_centroids:
        return []
        
    centroid_ids = list(known_centroids.keys())
    centroid_matrix = np.array([known_centroids[cid]['centroid'] for cid in centroid_ids])
    
    distances = custom_cosine_distances(target_embedding, centroid_matrix)[0]
    
    matches = []
    for i, cid in enumerate(centroid_ids):
        dist = distances[i]
        confidence = calculate_confidence(dist)
        if confidence > 0:
            matches.append({
                'cluster_id': cid,
                'name': known_centroids[cid]['name'],
                'confidence': round(confidence, 1),
                'distance': round(float(dist), 3)
            })
            
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches[:top_k]

def batch_suggest_names(unnamed_clusters_temp, centroid_matrix, centroid_names, threshold=0.45):
    """
    Batch calculates name suggestions for unnamed clusters against known centroids.
    Modifies unnamed_clusters_temp in place by adding 'suggested_name'.
    """
    if not unnamed_clusters_temp or centroid_matrix is None:
        for cluster_data, _ in unnamed_clusters_temp:
            cluster_data['suggested_name'] = None
        return
        
    unnamed_centroids = np.array([c[1] for c in unnamed_clusters_temp])
    all_distances = custom_cosine_distances(unnamed_centroids, centroid_matrix)
    
    for i, (cluster_data, _) in enumerate(unnamed_clusters_temp):
        best_idx = np.argmin(all_distances[i])
        best_dist = all_distances[i][best_idx]
        
        if best_dist < threshold:
            cluster_data['suggested_name'] = f"{centroid_names[best_idx]}?"
        else:
            cluster_data['suggested_name'] = None

def process_quicksort(face_data, embeddings_list, known_centroids):
    """
    Calculates distances for all unsorted faces against known centroids,
    and returns grouped results sorted by confidence.
    """
    centroid_ids = list(known_centroids.keys())
    centroid_matrix = np.array([known_centroids[cid]['centroid'] for cid in centroid_ids])
    face_matrix = np.array(embeddings_list)
    
    all_distances = custom_cosine_distances(face_matrix, centroid_matrix)
    
    groups = {}
    no_match = []
    
    for i, fd in enumerate(face_data):
        distances = all_distances[i]
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        
        confidence = calculate_confidence(best_dist)
        confidence = round(confidence, 1)
        
        if confidence > 5:
            cid = centroid_ids[best_idx]
            if cid not in groups:
                groups[cid] = {
                    'cluster_id': cid,
                    'name': known_centroids[cid]['name'],
                    'faces': []
                }
            groups[cid]['faces'].append({
                'id': fd['id'],
                'confidence': confidence
            })
        else:
            no_match.append({
                'id': fd['id'],
                'confidence': confidence
            })
            
    # Calculate averages and sort
    result_groups = []
    for gid, group in groups.items():
        avg_conf = sum(f['confidence'] for f in group['faces']) / len(group['faces'])
        group['avg_confidence'] = round(avg_conf, 1)
        group['faces'].sort(key=lambda x: x['confidence'], reverse=True)
        result_groups.append(group)
        
    result_groups.sort(key=lambda x: x['avg_confidence'], reverse=True)
    
    if no_match:
        result_groups.append({
            'cluster_id': None,
            'name': None,
            'faces': no_match,
            'avg_confidence': 0
        })
        
    return result_groups
