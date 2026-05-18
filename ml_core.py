import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

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

def cluster_faces(embeddings, eps=0.45, min_samples=3):
    """
    Runs DBSCAN on the embeddings and returns cluster labels.
    """
    if not len(embeddings):
        return []
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    return db.fit_predict(embeddings)

def get_top_matches(target_embedding, known_centroids, top_k=5):
    """
    Compares a target embedding against known centroids and returns the top matches.
    Returns: list of dicts with 'cluster_id', 'name', 'confidence', 'distance'.
    """
    if not known_centroids:
        return []
        
    centroid_ids = list(known_centroids.keys())
    centroid_matrix = np.array([known_centroids[cid]['centroid'] for cid in centroid_ids])
    
    distances = cosine_distances(target_embedding, centroid_matrix)[0]
    
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
    all_distances = cosine_distances(unnamed_centroids, centroid_matrix)
    
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
    
    all_distances = cosine_distances(face_matrix, centroid_matrix)
    
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
