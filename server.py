"""
Face Sorter Server - V3 Fresh Rewrite
======================================
Flask API backend for the Face Sorter application.

Endpoints:
    GET  /                     - Serve the main web app
    GET  /api/clusters         - Get paginated clusters (named/unnamed/unsorted)
    POST /api/cluster          - Run DBSCAN clustering
    POST /api/rename           - Rename/merge a cluster
    POST /api/face/<id>/move   - Move a face to another cluster
    POST /api/face/<id>/remove - Mark a face as noise (-1)
    GET  /api/thumbnail/<id>   - Get face thumbnail image
"""

import os
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import ml_core
from io import BytesIO

# Import database models
from database import Face, Image, Person, FaceThumbnail, BaseMetadata

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), 'metadata.db')
THUMB_DB_PATH = os.path.join(os.path.dirname(__file__), 'thumbnails.db')

# Create engines
engine_meta = create_engine(f'sqlite:///{DB_PATH}', echo=False)
engine_thumb = create_engine(f'sqlite:///{THUMB_DB_PATH}', echo=False)

SessionMeta = sessionmaker(bind=engine_meta)
SessionThumb = sessionmaker(bind=engine_thumb)

# Flask app
app = Flask(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_faces_df():
    """
    Fetch all faces with their embeddings and cluster info.
    Returns a list of dicts for easier manipulation.
    """
    from sqlalchemy.orm import joinedload
    
    session = SessionMeta()
    try:
        # Eager load the image relationship to avoid DetachedInstanceError
        faces = session.query(Face).options(joinedload(Face.image)).all()
        data = []
        for f in faces:
            data.append({
                'id': f.id,
                'image_id': f.image_id,
                'path': f.image.path if f.image else '',
                'cluster_id': f.cluster_id if f.cluster_id is not None else -1,
                'is_verified': f.is_verified or False,
                'embedding': np.frombuffer(f.embedding, dtype=np.float32) if f.embedding else None,
                'det_score': f.det_score or 0.0
            })
        
        # Filter out "Trash" (cluster_id = -2)
        data = [f for f in data if f['cluster_id'] != -2]
        
        return data
    finally:
        session.close()


def get_person_centroids():
    """
    Get all named people with their mean embeddings (centroids).
    Returns dict: {cluster_id: {'name': str, 'centroid': np.array, 'count': int}}
    """
    session = SessionMeta()
    try:
        people = session.query(Person).all()
        centroids = {}
        for p in people:
            if p.mean_embedding:
                centroids[p.cluster_id] = {
                    'name': p.name,
                    'centroid': np.frombuffer(p.mean_embedding, dtype=np.float32),
                    'count': p.face_count or 0
                }
        return centroids
    finally:
        session.close()


def suggest_name_for_cluster(cluster_faces, known_centroids, threshold=0.45):
    """
    Compare a cluster's average embedding to known centroids.
    Returns suggested name if distance < threshold, else None.
    """
    if not cluster_faces or not known_centroids:
        return None
    
    embeddings = [f['embedding'] for f in cluster_faces if f['embedding'] is not None]
    if not embeddings:
        return None
    
    cluster_centroid = ml_core.calculate_centroid(embeddings).reshape(1, -1)
    matches = ml_core.get_top_matches(cluster_centroid, known_centroids, top_k=1)
    
    if matches and matches[0]['distance'] < threshold:
        return f"{matches[0]['name']}?"
    return None


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main web application."""
    return render_template('index.html')


@app.route('/quicksort')
def quicksort_page():
    """Serve the Quick Sort page."""
    return render_template('quicksort.html')


@app.route('/api/quicksort')
def get_quicksort_data():
    """
    Analyze ALL unsorted faces and group them by best match.
    Returns groups sorted by average confidence (highest first).
    """
    session = SessionMeta()
    try:
        # Get all unsorted faces (cluster_id = -1)
        unsorted_faces = session.query(Face).filter_by(cluster_id=-1).all()
        
        if not unsorted_faces:
            return jsonify({'groups': [], 'total': 0})
        
        # Get known people centroids
        known_centroids = get_person_centroids()
        
        if not known_centroids:
            # No known people, return all as "no match"
            return jsonify({
                'groups': [{
                    'name': None,
                    'cluster_id': None,
                    'faces': [{'id': f.id, 'confidence': 0} for f in unsorted_faces],
                    'avg_confidence': 0
                }],
                'total': len(unsorted_faces)
            })
        
        # Collect all embeddings
        face_data = []
        embeddings_list = []
        for f in unsorted_faces:
            if f.embedding:
                face_data.append({'id': f.id})
                embeddings_list.append(np.frombuffer(f.embedding, dtype=np.float32))
        
        if not embeddings_list:
            return jsonify({'groups': [], 'total': 0})
            
        result_groups = ml_core.process_quicksort(face_data, embeddings_list, known_centroids)


        
        return jsonify({
            'groups': result_groups,
            'total': len(face_data)
        })
    finally:
        session.close()


@app.route('/api/quicksort/accept', methods=['POST'])
def quicksort_accept():
    """
    Accept a group of faces and assign them to a cluster.
    Request body: {cluster_id: int, face_ids: [int, ...]}
    """
    data = request.json or {}
    cluster_id = data.get('cluster_id')
    face_ids = data.get('face_ids', [])
    
    if not cluster_id or not face_ids:
        return jsonify({'error': 'Missing cluster_id or face_ids'}), 400
    
    session = SessionMeta()
    try:
        # Update all specified faces
        session.query(Face).filter(Face.id.in_(face_ids)).update(
            {Face.cluster_id: cluster_id, Face.is_verified: True},
            synchronize_session=False
        )
        
        # Update person's face count and centroid
        person = session.query(Person).filter_by(cluster_id=cluster_id).first()
        if person:
            # Get all faces now in this cluster to recalculate centroid
            cluster_faces = session.query(Face).filter_by(cluster_id=cluster_id).all()
            embeddings = [np.frombuffer(f.embedding, dtype=np.float32) for f in cluster_faces if f.embedding]
            if embeddings:
                new_centroid = ml_core.calculate_centroid(embeddings)
                person.mean_embedding = new_centroid.tobytes()
                person.face_count = len(embeddings)
        
        session.commit()
        return jsonify({'status': 'ok', 'count': len(face_ids)})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/db')
def db_viewer():
    """Serve the database viewer page."""
    return render_template('db.html')


@app.route('/api/db/stats')
def db_stats():
    """Get database statistics."""
    session = SessionMeta()
    try:
        images = session.query(Image).count()
        faces = session.query(Face).count()
        people = session.query(Person).count()
        return jsonify({
            'images': images,
            'faces': faces,
            'people': people
        })
    finally:
        session.close()


@app.route('/api/db/table/<table_name>')
def db_table(table_name):
    """Get paginated table data."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    
    # Map table names to models
    tables = {
        'images': Image,
        'faces': Face,
        'people': Person
    }
    
    if table_name not in tables:
        return jsonify({'error': 'Invalid table'}), 400
    
    model = tables[table_name]
    session = SessionMeta()
    
    try:
        total = session.query(model).count()
        offset = (page - 1) * per_page
        rows = session.query(model).offset(offset).limit(per_page).all()
        
        # Get column names
        columns = [c.name for c in model.__table__.columns]
        
        # Convert rows to dicts
        data = []
        for row in rows:
            row_dict = {}
            for col in columns:
                val = getattr(row, col)
                # Handle binary data
                if isinstance(val, bytes):
                    row_dict[col] = f"[BLOB {len(val)} bytes]"
                else:
                    row_dict[col] = val
            data.append(row_dict)
        
        return jsonify({
            'columns': columns,
            'rows': data,
            'total': total,
            'page': page,
            'per_page': per_page
        })
    finally:
        session.close()


@app.route('/api/clusters')
def get_clusters():
    """
    Get paginated clusters organized into named, unnamed, and unsorted.
    
    Query Parameters:
        page_named (int): Page number for verified/named clusters
        page_unnamed (int): Page number for workbench clusters
        per_page (int): Items per page
        preview_size (int): Number of face thumbnails to include per cluster
        search_named (str): Filter named clusters by name
        search_unnamed (str): Filter unnamed clusters
    """
    # Parse parameters
    page_named = int(request.args.get('page_named', 1))
    page_unnamed = int(request.args.get('page_unnamed', 1))
    per_page = int(request.args.get('per_page', 20))
    preview_size = int(request.args.get('preview_size', 24))
    search_named = request.args.get('search_named', '').lower().strip()
    search_unnamed = request.args.get('search_unnamed', '').lower().strip()
    only_suggestions = request.args.get('only_suggestions', 'false') == 'true'
    
    # Fetch all data
    faces = get_all_faces_df()
    
    if not faces:
        return jsonify({
            'named_clusters': [],
            'unnamed_clusters': [],
            'unsorted_faces': [],
            'pagination': {
                'named': {'page': 1, 'total': 0, 'count': 0},
                'unnamed': {'page': 1, 'total': 0, 'count': 0},
                'per_page': per_page
            },
            'unsorted_count': 0
        })
    
    # Get known people
    session = SessionMeta()
    try:
        people = {p.cluster_id: p.name for p in session.query(Person).all()}
    finally:
        session.close()
    
    # Known centroids for suggestions
    known_centroids = get_person_centroids()
    
    # Group faces by cluster
    clusters = {}
    unsorted = []
    
    for f in faces:
        cid = f['cluster_id']
        if cid == -1:
            unsorted.append({'id': f['id'], 'path': f['path']})
        else:
            if cid not in clusters:
                clusters[cid] = {
                    'id': cid,
                    'faces': [],
                    'is_verified': False,
                    'name': people.get(cid, '')
                }
            clusters[cid]['faces'].append(f)
            if f['is_verified']:
                clusters[cid]['is_verified'] = True
    
    # Build cluster list with metadata
    named_list = []
    unnamed_list = []
    unnamed_clusters_temp = []  # Store for batch suggestion
    
    # Pre-compute centroid matrix once for all suggestions
    centroid_ids = list(known_centroids.keys()) if known_centroids else []
    centroid_matrix = np.array([known_centroids[cid]['centroid'] for cid in centroid_ids]) if centroid_ids else None
    centroid_names = [known_centroids[cid]['name'] for cid in centroid_ids] if centroid_ids else []
    
    for cid, cluster in clusters.items():
        cluster_data = {
            'id': cid,
            'name': cluster['name'],
            'count': len(cluster['faces']),
            'faces': [{'id': f['id'], 'path': f['path']} for f in cluster['faces'][:preview_size]],
            'is_verified': cluster['is_verified']
        }
        
        # Determine category
        if cluster['name'] or cluster['is_verified']:
            named_list.append(cluster_data)
        else:
            # Calculate cluster centroid for batch suggestion
            embeddings = [f['embedding'] for f in cluster['faces'] if f.get('embedding') is not None]
            if embeddings and centroid_matrix is not None:
                cluster_centroid = ml_core.calculate_centroid(embeddings)
                unnamed_clusters_temp.append((cluster_data, cluster_centroid))
            else:
                cluster_data['suggested_name'] = None
                unnamed_list.append(cluster_data)
    
    # Batch calculate suggestions for all unnamed clusters at once
    if unnamed_clusters_temp and centroid_matrix is not None:
        ml_core.batch_suggest_names(unnamed_clusters_temp, centroid_matrix, centroid_names)
        for cluster_data, _ in unnamed_clusters_temp:
            unnamed_list.append(cluster_data)
    

    
    # Sort by count (descending)
    named_list.sort(key=lambda x: x['count'], reverse=True)
    unnamed_list.sort(key=lambda x: x['count'], reverse=True)
    
    # Apply search filters
    if search_named:
        named_list = [c for c in named_list if search_named in (c['name'] or '').lower()]
    if search_unnamed:
        unnamed_list = [c for c in unnamed_list if search_unnamed in (c.get('suggested_name') or '').lower()]
        
    if only_suggestions:
        unnamed_list = [c for c in unnamed_list if c.get('suggested_name')]
    
    # Calculate pagination
    total_named = len(named_list)
    total_unnamed = len(unnamed_list)
    total_pages_named = max(1, (total_named + per_page - 1) // per_page)
    total_pages_unnamed = max(1, (total_unnamed + per_page - 1) // per_page)
    
    # Slice for current page
    start_n = (page_named - 1) * per_page
    end_n = start_n + per_page
    start_u = (page_unnamed - 1) * per_page
    end_u = start_u + per_page
    
    paginated_named = named_list[start_n:end_n]
    paginated_unnamed = unnamed_list[start_u:end_u]
    
    # Run heavy ML predictions ONLY on the faces that are actually being shown on this page
    for c in paginated_named + paginated_unnamed:
        # Grab embeddings from the original clusters dictionary, since c['faces'] only contains id and path
        embeddings = [f['embedding'] for f in clusters[c['id']]['faces'] if f.get('embedding') is not None]
        if embeddings:
            c['race'] = ml_core.predict_race(embeddings)
        else:
            c['race'] = "Unknown"
    
    return jsonify({
        'named_clusters': paginated_named,
        'unnamed_clusters': paginated_unnamed,
        'unsorted_faces': unsorted[:preview_size],
        'unsorted_total': len(unsorted),
        'pagination': {
            'named': {'page': page_named, 'total': total_pages_named, 'count': total_named},
            'unnamed': {'page': page_unnamed, 'total': total_pages_unnamed, 'count': total_unnamed},
            'per_page': per_page
        }
    })


@app.route('/api/cluster', methods=['POST'])
def run_clustering():
    """
    Run DBSCAN clustering on all unverified faces.
    
    Request Body:
        eps (float): DBSCAN epsilon parameter (default 0.45)
        min_samples (int): DBSCAN min_samples parameter (default 3)
    """
    data = request.json or {}
    eps = float(data.get('eps', 0.45))
    min_samples = int(data.get('min_samples', 3))
    
    # Get all faces
    faces = get_all_faces_df()
    
    if not faces:
        return jsonify({'status': 'ok', 'clusters': 0, 'message': 'No faces to cluster'})
    
    # Filter to only unverified faces with valid embeddings
    unverified = [f for f in faces if not f['is_verified'] and f['embedding'] is not None]
    
    if not unverified:
        return jsonify({'status': 'ok', 'clusters': 0, 'message': 'No unverified faces'})
    
    # Build embedding matrix
    face_ids = [f['id'] for f in unverified]
    embeddings = np.array([f['embedding'] for f in unverified])
    
    # Run DBSCAN with cosine distance on full 512D embeddings
    labels = ml_core.cluster_faces(embeddings, eps=eps, min_samples=min_samples)
    
    # Get the next available cluster ID
    session = SessionMeta()
    try:
        max_cluster = session.query(Face.cluster_id).order_by(Face.cluster_id.desc()).first()
        next_id = (max_cluster[0] or 0) + 1 if max_cluster and max_cluster[0] and max_cluster[0] > 0 else 1
        
        # Remap labels: -1 stays as -1, others get new positive IDs
        label_map = {}
        new_cluster_count = 0
        
        for label in set(labels):
            if label == -1:
                label_map[-1] = -1
            else:
                label_map[label] = next_id
                next_id += 1
                new_cluster_count += 1
        
        # Update database
        for face_id, label in zip(face_ids, labels):
            face = session.query(Face).filter_by(id=face_id).first()
            if face:
                face.cluster_id = label_map[label]
        
        session.commit()
        
        return jsonify({
            'status': 'ok',
            'clusters': new_cluster_count,
            'noise': int(np.sum(labels == -1)),
            'message': f'Created {new_cluster_count} clusters, {np.sum(labels == -1)} unsorted'
        })
    except Exception as e:
        session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        session.close()


@app.route('/api/rename', methods=['POST'])
def rename_cluster():
    """
    Rename a cluster or merge it into an existing person.
    Updates the person's centroid using weighted average.
    
    Request Body:
        id (int): Cluster ID to rename
        name (str): New name for the cluster
    """

    
    data = request.json or {}
    cluster_id = data.get('id')
    new_name = data.get('name', '').strip()
    
    if not cluster_id or not new_name:
        return jsonify({'error': 'Missing id or name'}), 400
    
    session = SessionMeta()
    try:
        # Check if this is a simple rename of an existing Person (even if empty)
        existing_person = session.query(Person).filter_by(cluster_id=cluster_id).first()
        
        # Get all faces in this cluster
        faces = session.query(Face).filter_by(cluster_id=cluster_id).all()
        
        if not faces and not existing_person:
            return jsonify({'error': 'Cluster/Person not found'}), 404
            
        # If we have faces, we do the full merge/move logic
        if faces:
            # Calculate the cluster's mean embedding
            embeddings = [np.frombuffer(f.embedding, dtype=np.float32) for f in faces if f.embedding]
            if not embeddings:
                 # Should not happen if faces exist, but safe fallback
                if existing_person:
                    existing_person.name = new_name
                    session.commit()
                    return jsonify({'status': 'ok', 'message': 'Renamed empty person'})
                return jsonify({'error': 'No valid embeddings'}), 400
            
            cluster_mean = ml_core.calculate_centroid(embeddings)
            cluster_count = len(faces)
            
            # Check if target name already exists (Merge)
            target_person = session.query(Person).filter_by(name=new_name).first()
            
            target_cluster_id = None
            
            if target_person:
                # Merge into existing person
                if target_person.cluster_id == cluster_id:
                     # Same person, nothing to do
                     return jsonify({'status': 'ok'})
                     
                old_mean = np.frombuffer(target_person.mean_embedding, dtype=np.float32) if target_person.mean_embedding else cluster_mean
                old_count = target_person.face_count or 1
                
                # Weighted average
                new_mean = ml_core.calculate_weighted_centroid(old_mean, old_count, cluster_mean, cluster_count)
                target_person.mean_embedding = new_mean.astype(np.float32).tobytes()
                target_person.face_count = old_count + cluster_count
                target_cluster_id = target_person.cluster_id
                
                # Delete the old person entry if it existed
                if existing_person and existing_person.cluster_id != target_cluster_id:
                    session.delete(existing_person)
            else:
                # Rename current person or Create new
                if existing_person:
                    existing_person.name = new_name
                    # Embeddings update shouldn't strictly be necessary if just renaming, 
                    # but good to ensure consistency
                    existing_person.mean_embedding = cluster_mean.astype(np.float32).tobytes()
                    existing_person.face_count = cluster_count
                    target_cluster_id = cluster_id
                else:
                    # Creating new person from unnamed cluster
                    person = Person(
                        cluster_id=cluster_id,
                        name=new_name,
                        mean_embedding=cluster_mean.astype(np.float32).tobytes(),
                        face_count=cluster_count
                    )
                    session.add(person)
                    target_cluster_id = cluster_id
            
            # Update all faces
            for face in faces:
                face.cluster_id = target_cluster_id
                face.is_verified = True
                
        else:
            # No faces, just renaming an empty Person
            if existing_person:
                # Check if target name collision
                target_person = session.query(Person).filter_by(name=new_name).first()
                if target_person:
                    # Merge empty person into target? Actually just delete empty person
                    session.delete(existing_person) 
                    # Faces are 0, so nothing to move.
                else:
                    existing_person.name = new_name
            
            target_cluster_id = cluster_id # Keep same ID
            cluster_count = 0

        session.commit()
        
        return jsonify({
            'status': 'ok',
            'cluster_id': target_cluster_id,
            'name': new_name,
            'face_count': cluster_count
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/api/face/<int:face_id>/move', methods=['POST'])
def move_face(face_id):
    """
    Move a face to a different cluster.
    
    Request Body:
        target_cluster_id (int): Cluster ID to move the face to
        target_name (str): If provided, create new cluster with this name
    """
    data = request.json or {}
    target_cluster_id = data.get('target_cluster_id')
    target_name = data.get('target_name', '').strip()
    
    session = SessionMeta()
    try:
        face = session.query(Face).filter_by(id=face_id).first()
        if not face:
            return jsonify({'error': 'Face not found'}), 404
        
        if target_name:
            # Create or find person by name
            person = session.query(Person).filter_by(name=target_name).first()
            if person:
                target_cluster_id = person.cluster_id
            else:
                # Get next cluster ID
                max_cluster = session.query(Face.cluster_id).order_by(Face.cluster_id.desc()).first()
                target_cluster_id = (max_cluster[0] or 0) + 1 if max_cluster and max_cluster[0] and max_cluster[0] > 0 else 1
                
                # Create new person
                embedding = np.frombuffer(face.embedding, dtype=np.float32) if face.embedding else None
                person = Person(
                    cluster_id=target_cluster_id,
                    name=target_name,
                    mean_embedding=embedding.tobytes() if embedding is not None else None,
                    face_count=1
                )
                session.add(person)
        
        if target_cluster_id is None:
            return jsonify({'error': 'No target specified'}), 400
        
        # Move the face
        face.cluster_id = target_cluster_id
        face.is_verified = True
        
        session.commit()
        
        return jsonify({
            'status': 'ok',
            'face_id': face_id,
            'new_cluster_id': target_cluster_id
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/api/face/<int:face_id>/race')
def get_face_race(face_id):
    session = SessionMeta()
    try:
        f = session.query(Face).get(face_id)
        if f and f.embedding:
            embedding = np.frombuffer(f.embedding, dtype=np.float32)
            race = ml_core.predict_race([embedding])
            return jsonify({'status': 'ok', 'race': race})
        return jsonify({'status': 'error', 'error': 'Face not found'})
    finally:
        session.close()

@app.route('/api/face/<int:face_id>/matches')
def get_face_matches(face_id):
    """
    Get top matches for an unsorted face compared to known people.
    Returns list of matches with confidence percentages.
    """
    session = SessionMeta()
    try:
        face = session.query(Face).filter_by(id=face_id).first()
        if not face or not face.embedding:
            return jsonify({'matches': []})
        
        # Get known people centroids
        known_centroids = get_person_centroids()
        print(f"[DEBUG] match: Found {len(known_centroids)} known people")
        if not known_centroids:
            return jsonify({'matches': []})
        
        # Get face embedding
        face_embedding = np.frombuffer(face.embedding, dtype=np.float32).reshape(1, -1)
        
        matches = ml_core.get_top_matches(face_embedding, known_centroids, top_k=5)
        
        # Return top 5 matches
        return jsonify({'matches': matches})
    finally:
        session.close()


@app.route('/api/cluster/<int:cluster_id>/matches')
def get_cluster_matches(cluster_id):
    """
    Get top matches for an UNNAMED cluster compared to known people.
    Calculates the cluster's mean embedding and compares it.
    Returns list of matches with confidence percentages.
    """
    session = SessionMeta()
    try:
        # Get all faces in the cluster to calculate mean embedding
        faces = session.query(Face).filter_by(cluster_id=cluster_id).all()
        embeddings = [np.frombuffer(f.embedding, dtype=np.float32) for f in faces if f.embedding]
        
        if not embeddings:
            return jsonify({'matches': []})
            
        cluster_centroid = ml_core.calculate_centroid(embeddings).reshape(1, -1)
        
        # Get known people centroids
        known_centroids = get_person_centroids()
        if not known_centroids:
            return jsonify({'matches': []})
        
        matches = ml_core.get_top_matches(cluster_centroid, known_centroids, top_k=5)
        
        return jsonify({'matches': matches})
    finally:
        session.close()


@app.route('/api/face/<int:face_id>/remove', methods=['POST'])
def remove_face(face_id):
    """
    Mark a face as noise/unsorted (cluster_id = -1).
    """
    session = SessionMeta()
    try:
        face = session.query(Face).filter_by(id=face_id).first()
        if not face:
            return jsonify({'error': 'Face not found'}), 404
        
        face.cluster_id = -1
        face.is_verified = False
        session.commit()
        
        return jsonify({'status': 'ok', 'face_id': face_id})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/api/face/<int:face_id>/trash', methods=['POST'])
def trash_face(face_id):
    """
    Mark a face as trash/dismissed (cluster_id = -2).
    It will be hidden from UI but kept in DB to prevent re-indexing.
    """
    session = SessionMeta()
    try:
        face = session.query(Face).filter_by(id=face_id).first()
        if not face:
            return jsonify({'error': 'Face not found'}), 404
        
        face.cluster_id = -2
        face.is_verified = False # It's not verified if it's trash
        session.commit()
        
        return jsonify({'status': 'ok', 'face_id': face_id})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/api/cluster/<int:cluster_id>/trash', methods=['POST'])
def trash_cluster(cluster_id):
    """
    Mark ALL faces in a cluster as trash (cluster_id = -2).
    """
    session = SessionMeta()
    try:
        # Update all faces in this cluster
        session.query(Face).filter_by(cluster_id=cluster_id).update(
            {Face.cluster_id: -2, Face.is_verified: False},
            synchronize_session=False
        )
        session.commit()
        return jsonify({'status': 'ok', 'cluster_id': cluster_id})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/api/cluster/<cluster_id>/faces')
def get_cluster_faces(cluster_id):
    """
    Get all faces in a specific cluster.
    Accepts string to handle negative IDs like -1.
    """
    try:
        cid = int(cluster_id)
    except ValueError:
        return jsonify({'faces': []}), 400
    
    session = SessionMeta()
    try:
        from sqlalchemy.orm import joinedload
        faces = session.query(Face).options(joinedload(Face.image)).filter_by(cluster_id=cid).all()
        result = []
        for f in faces:
            result.append({
                'id': f.id,
                'path': f.image.path if f.image else ''
            })
        return jsonify({'faces': result})
    finally:
        session.close()


@app.route('/api/thumbnail/<int:face_id>')
def get_thumbnail(face_id):
    """Get a face thumbnail image."""
    session = SessionThumb()
    try:
        thumb = session.query(FaceThumbnail).filter_by(face_id=face_id).first()
        if not thumb or not thumb.thumbnail:
            return '', 404
        
        return send_file(
            BytesIO(thumb.thumbnail),
            mimetype='image/jpeg',
            max_age=86400  # Cache for 24 hours
        )
    finally:
        session.close()


@app.route('/api/image/<int:face_id>')
def get_original_image(face_id):
    """Get the original full-size image for a face."""
    session = SessionMeta()
    try:
        face = session.query(Face).filter_by(id=face_id).first()
        if not face or not face.image:
            return '', 404
        
        image_path = face.image.path
        if not os.path.exists(image_path):
            return '', 404
        
        # Determine mimetype from extension
        ext = os.path.splitext(image_path)[1].lower()
        mimetypes = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mimetype = mimetypes.get(ext, 'image/jpeg')
        
        return send_file(image_path, mimetype=mimetype)
    finally:
        session.close()


@app.route('/api/unsorted/all')
def get_all_unsorted():
    """Get all unsorted faces (cluster_id = -1)."""
    session = SessionMeta()
    try:
        from sqlalchemy.orm import joinedload
        faces = session.query(Face).options(joinedload(Face.image)).filter_by(cluster_id=-1).all()
        result = []
        for f in faces:
            result.append({
                'id': f.id,
                'path': f.image.path if f.image else ''
            })
        return jsonify({'faces': result, 'count': len(result)})
    finally:
        session.close()


@app.route('/api/people')
def get_people():
    """Get list of all named people for context menu, with face counts."""
    session = SessionMeta()
    try:
        from sqlalchemy import func
        
        # Get all people
        people = session.query(Person).order_by(Person.name).all()
        
        # Get counts for all valid clusters
        # query: SELECT cluster_id, COUNT(*) FROM faces GROUP BY cluster_id
        counts = session.query(Face.cluster_id, func.count(Face.id))\
            .filter(Face.cluster_id != None)\
            .group_by(Face.cluster_id).all()
            
        count_map = {c[0]: c[1] for c in counts}
        
        result = []
        for p in people:
            result.append({
                'cluster_id': p.cluster_id, 
                'name': p.name,
                'count': count_map.get(p.cluster_id, 0)
            })
            
        # Sort by count (descending), then name (ascending)
        result.sort(key=lambda x: (-x['count'], x['name']))

        return jsonify({ 'people': result })
    finally:
        session.close()

@app.route('/api/person/<int:cluster_id>', methods=['DELETE'])
def delete_person_name(cluster_id):
    """Delete a person's name (un-verify the cluster)."""
    session = SessionMeta()
    try:
        person = session.query(Person).filter_by(cluster_id=cluster_id).first()
        if not person:
            return jsonify({'error': 'Person not found'}), 404
            
        name = person.name
        session.delete(person)
        session.commit()
        return jsonify({'status': 'ok', 'message': f'Deleted {name}'})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
