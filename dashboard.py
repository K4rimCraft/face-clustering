import streamlit as st
import pandas as pd
import numpy as np
import base64
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from streamlit_image_select import image_select

# Local imports
from database import Image, Face, FaceThumbnail, Person, METADATA_DB_NAME, THUMBNAILS_DB_NAME

# --- Config ---
st.set_page_config(layout="wide", page_title="Face Sorter AI")

# --- CSS ---
st.markdown("""
<style>
    .stTextInput input {
        font-weight: bold;
        font-size: 1.2em;
        color: #4A90E2;
    }
    div[data-testid="stContainer"] {
        background-color: #f9f9f910;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Connection (Cached) ---
@st.cache_resource
def get_sessions():
    engine_meta = create_engine(f'sqlite:///{METADATA_DB_NAME}', echo=False)
    engine_thumb = create_engine(f'sqlite:///{THUMBNAILS_DB_NAME}', echo=False)
    
    with engine_meta.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    with engine_thumb.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")

    SessionMeta = sessionmaker(bind=engine_meta)
    SessionThumb = sessionmaker(bind=engine_thumb)
    return SessionMeta, SessionThumb

SessionMeta, SessionThumb = get_sessions()

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    """Loads all embeddings and face metadata."""
    session = SessionMeta()
    
    query = session.query(
        Face.id, 
        Face.embedding, 
        Face.image_id, 
        Face.cluster_id, 
        Image.path
    ).join(Image, Face.image_id == Image.id)
    
    results = query.all()
    session.close()
    
    if not results:
        return pd.DataFrame(), np.array([])
        
    df = pd.DataFrame([{
        'id': r.id, 
        'embedding': r.embedding, 
        'image_id': r.image_id, 
        'cluster_id': r.cluster_id,
        'path': r.path
    } for r in results])
    
    embeddings = np.array([np.frombuffer(b, dtype=np.float32) for b in df['embedding']])
    
    return df, embeddings

@st.cache_data
def load_names():
    session = SessionMeta()
    results = session.query(Person).all()
    session.close()
    return {p.cluster_id: p.name for p in results}

# --- Clustering Logic ---
def run_clustering(df, embeddings, eps, min_samples, use_pca=True):
    if len(embeddings) == 0:
        return df
    
    X = embeddings
    if use_pca and len(embeddings) > 100:
        n_components = min(128, len(embeddings))
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = clusterer.fit_predict(X)
    
    df['new_cluster'] = labels
    return df

# --- Thumbnail Helper ---
def get_thumbnail_b64(face_id):
    """Returns base64 string for HTML display."""
    session = SessionThumb()
    result = session.query(FaceThumbnail.thumbnail).filter_by(face_id=face_id).first()
    session.close()
    if result and result[0]:
        return f"data:image/jpeg;base64,{base64.b64encode(result[0]).decode('utf-8')}"
    return None

# --- Main UI ---
def main():
    if 'expanded_clusters' not in st.session_state:
        st.session_state['expanded_clusters'] = set()
    if 'view_image' not in st.session_state:
        st.session_state['view_image'] = None

    # --- Full Screen View Logic ---
    if st.session_state['view_image']:
        img_path = st.session_state['view_image']
        
        # Header
        col_back, col_title = st.columns([1, 10])
        with col_back:
            if st.button("⬅ Back", use_container_width=True):
                st.session_state['view_image'] = None
                st.rerun()
        with col_title:
             st.subheader(f"Viewing: {img_path}")
             
        # Full Image
        st.image(img_path, use_container_width=True)
        return

    st.title("Face Sorter Dashboard")
    
    with st.spinner("Loading Database..."):
        df, embeddings = load_data()
        
    names_map = load_names()
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")
    page_size = st.sidebar.number_input("People per page", 5, 200, 20)
    
    st.sidebar.subheader("Clustering")
    eps = st.sidebar.slider("EPS (Likeness)", 0.0, 1.0, 0.45, 0.01)
    min_samples = st.sidebar.slider("Min Faces", 1, 10, 3)
    
    if st.sidebar.button("Re-Cluster", type="primary"):
        with st.spinner("Clustering..."):
            df = run_clustering(df, embeddings, eps, min_samples)
            st.session_state['clustered_df'] = df
            st.session_state['expanded_clusters'] = set()
            st.success("Done!")

    if 'clustered_df' in st.session_state:
        df = st.session_state['clustered_df']
    else:
        df = run_clustering(df, embeddings, eps, min_samples)
        st.session_state['clustered_df'] = df
        
    group_col = 'new_cluster'
    unique_clusters = df[group_col].value_counts().index
    sorted_clusters = [c for c in unique_clusters if c != -1]
    
    # --- Pagination ---
    total_pages = max(1, len(sorted_clusters)//page_size + 1)
    page = st.number_input("Page", 1, total_pages, 1)
    
    start_idx = (page - 1) * page_size
    current_batch = sorted_clusters[start_idx : start_idx + page_size]
    
    # --- Rendering Grid ---
    for cluster_id in current_batch:
        cluster_df = df[df[group_col] == cluster_id]
        current_name = names_map.get(cluster_id, f"Person {cluster_id}")
        
        with st.container(border=True):
            col_info, col_grid = st.columns([1, 4])
            
            with col_info:
                st.subheader("👤 Identity")
                new_name = st.text_input("Name", value=(current_name if "Person" not in current_name else ""), key=f"name_{cluster_id}", placeholder="Name this person")
                if new_name and new_name != current_name:
                    session = SessionMeta()
                    person = session.query(Person).filter_by(cluster_id=int(cluster_id)).first()
                    if not person:
                        person = Person(cluster_id=int(cluster_id), name=new_name)
                        session.add(person)
                    else:
                        person.name = new_name
                    session.commit()
                    session.close()
                st.caption(f"{len(cluster_df)} photos")

            with col_grid:
                # Prepare images for selection
                is_expanded = cluster_id in st.session_state['expanded_clusters']
                limit = len(cluster_df) if is_expanded else 24
                
                faces_to_show = cluster_df.head(limit)
                
                # We need lists for image_select
                images_b64 = []
                captions = []
                indices = []
                
                for i, (_, row) in enumerate(faces_to_show.iterrows()):
                    b64 = get_thumbnail_b64(row['id'])
                    if b64:
                        images_b64.append(b64)
                        captions.append("") # Clean look
                        indices.append(row['path'])
                
                if images_b64:
                    # Render clickable grid
                    selected_path = image_select(
                        label="",
                        images=images_b64,
                        captions=captions,
                        use_container_width=False,
                        key=f"select_{cluster_id}",
                        return_value="index" # returns index 0, 1, 2...
                    )
                    
                    # Check if selection changed
                    if selected_path != -1:
                        # Map index back to path
                        real_path = indices[selected_path]
                        # Only trigger if it's a new interaction
                        if st.session_state.get(f"last_click_{cluster_id}") != selected_path:
                            st.session_state[f"last_click_{cluster_id}"] = selected_path
                            st.session_state['view_image'] = real_path
                            st.rerun()

                # Load More
                if len(cluster_df) > 24 and not is_expanded:
                     remaining = len(cluster_df) - 24
                     if st.button(f"⬇️ Load {remaining} more", key=f"more_{cluster_id}"):
                         st.session_state['expanded_clusters'].add(cluster_id)
                         st.rerun()

if __name__ == "__main__":
    main()
