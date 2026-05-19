import sys
from sqlalchemy.orm import sessionmaker
from database import get_db_engines, Face, Person

def reset_clustering_data():
    """
    Resets all clustering, labels, and verified data from the database.
    This does NOT delete images or extracted face embeddings.
    """
    engine_meta, _ = get_db_engines()
    Session = sessionmaker(bind=engine_meta)
    session = Session()

    print("Connecting to database...")
    try:
        # 1. Delete all saved people/labels
        deleted_people = session.query(Person).delete()
        print(f"Deleted {deleted_people} named clusters (People).")

        # 2. Reset all faces back to unsorted
        faces_updated = session.query(Face).update({
            "cluster_id": -1,
            "is_verified": False
        })
        print(f"Reset {faces_updated} faces back to Unsorted (-1).")

        # Commit changes
        session.commit()
        print("\n✅ Successfully reset all clustering data!")
        print("You can now refresh the web page or run DBSCAN clustering again.")
    except Exception as e:
        session.rollback()
        print(f"\n❌ Error resetting data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    reset_clustering_data()
