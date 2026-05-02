import os
import shutil
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Image, Face, Person, METADATA_DB_NAME
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

def get_session():
    engine = create_engine(f'sqlite:///{METADATA_DB_NAME}')
    Session = sessionmaker(bind=engine)
    return Session()

def safe_filename(name):
    """Sanitize directory names."""
    return "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()

def export_photos(output_dir, mode='link', unsorted_dir=None):
    session = get_session()
    
    # 1. Get all named people
    console.print("[bold blue]Fetching named clusters...[/bold blue]")
    named_people = session.query(Person).filter(Person.name != None).all()
    
    cluster_map = {p.cluster_id: safe_filename(p.name) for p in named_people}
    console.print(f"Found [green]{len(cluster_map)}[/green] named people.")
    
    # 2. Get all images and their clustered faces
    # We want a map of {image_id: [cluster_names]}
    # An image can belong to multiple clusters (people)
    
    image_destinations = {} # {image_path: set(folder_names)}
    
    query = session.query(Face.cluster_id, Image.path).\
        join(Image, Face.image_id == Image.id)
        
    total_faces = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Mapping faces to folders...", total=None)
        
        for cluster_id, path in query.yield_per(1000):
            total_faces += 1
            folder_name = None
            
            if cluster_id in cluster_map:
                folder_name = cluster_map[cluster_id]
            elif unsorted_dir and cluster_id == -1:
                folder_name = "_Unsorted"
            elif unsorted_dir:
                folder_name = "_Unknown_Clusters"
                
            if folder_name:
                if path not in image_destinations:
                    image_destinations[path] = set()
                image_destinations[path].add(folder_name)
                
    console.print(f"Mapped [bold]{total_faces}[/bold] faces to distinct destination sets.")
    
    # 3. Execution
    success_count = 0
    fail_count = 0
    
    # Create Base Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with Progress(
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.description}"),
    ) as progress:
        # Iterate over specific images effectively
        # image_destinations = { 'C:/img1.jpg': {'Mom', 'Dad'} }
        
        task = progress.add_task("Exporting...", total=len(image_destinations))
        
        for src_path, folders in image_destinations.items():
            if not folders:
                progress.advance(task)
                continue
                
            sorted_folders = sorted(list(folders)) # Deterministic order
            primary_folder = sorted_folders[0]
            secondary_folders = sorted_folders[1:]
            
            # --- PRIMARY ACTION ---
            # Moves or Copies the file to the FIRST location
            
            # Determine primary target
            target_dir = os.path.join(output_dir, primary_folder) if primary_folder not in ["_Unsorted", "_Unknown_Clusters"] else os.path.join(unsorted_dir, primary_folder)
            if primary_folder == "_Unsorted" and unsorted_dir: target_dir = unsorted_dir
            
            if not os.path.exists(target_dir): os.makedirs(target_dir, exist_ok=True)
            
            fname = os.path.basename(src_path)
            primary_dest = os.path.join(target_dir, fname)
            
            # Collision handling
            counter = 1
            base_name, ext = os.path.splitext(fname)
            while os.path.exists(primary_dest):
                if os.path.samefile(src_path, primary_dest) if os.path.exists(src_path) else False: break
                primary_dest = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            current_location = src_path
            
            try:
                # Perform the main Move/Link/Copy logic
                # STRATEGY: MOVE to Primary, LINK to Secondaries
                mode = 'move' # Enforce move mode
                final_action_taken = False
                
                # Check if src exists (it might have been moved in a previous run or by another process)
                if not os.path.exists(src_path):
                    # Check if it was already moved to the primary dest in a previous run?
                    if os.path.exists(primary_dest):
                        current_location = primary_dest
                        # Already moved, just ensure DB matches
                        final_action_taken = True 
                    else:
                        # Totally missing
                        # console.print(f"[red]Missing source:[/red] {src_path}")
                        fail_count += 1
                        progress.advance(task)
                        continue
                else:
                    # Source exists
                    
                    # Check if we are trying to move the file to itself (Already sorted)
                    is_same_file = False
                    try:
                        if os.path.exists(primary_dest) and os.path.samefile(src_path, primary_dest):
                            is_same_file = True
                    except OSError:
                        pass

                    if is_same_file:
                        # console.print(f"[grey50]Skipping {fname} (Already sorted)[/grey50]")
                        current_location = primary_dest
                        final_action_taken = True
                    else:
                        # MOVE it
                        try:
                            shutil.move(src_path, primary_dest)
                            current_location = primary_dest
                            final_action_taken = True
                        except OSError as e:
                            # Fallback for cross-fs move if shutil fails
                            shutil.copy2(src_path, primary_dest)
                            os.remove(src_path)
                            current_location = primary_dest
                            final_action_taken = True
                
                if final_action_taken:
                    # UPDATE DB
                    session_update = get_session()
                    img_row = session_update.query(Image).filter_by(path=src_path).first()
                    if img_row:
                        img_row.path = primary_dest
                        session_update.commit()
                    session_update.close()

                success_count += 1
                
                # --- SECONDARY ACTIONS ---
                # For Mom(Done) -> Dad(Next).
                # We LINK from the NEW location to save space.
                
                for folder in secondary_folders:
                    sec_target_dir = os.path.join(output_dir, folder)
                    if folder == "_Unsorted" and unsorted_dir: sec_target_dir = unsorted_dir
                    
                    if not os.path.exists(sec_target_dir): os.makedirs(sec_target_dir, exist_ok=True)
                    
                    sec_dest = os.path.join(sec_target_dir, os.path.basename(primary_dest))
                    
                    # Collision
                    counter = 1
                    while os.path.exists(sec_dest):
                        if os.path.exists(current_location) and os.path.samefile(current_location, sec_dest): break
                        sec_dest = os.path.join(sec_target_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                        
                    # SYMLINK (Shortcut)
                    # Note: Requires Admin or Developer Mode on Windows
                    try:
                        # os.symlink(target, link_name)
                        # We use absolute path for the target to be safe
                        os.symlink(current_location, sec_dest)
                    except OSError as e:
                        # "A required privilege is not held by the client" (WinError 1314)
                        console.print(f"[red]Symlink failed (Run as Admin!): {e}[/red]")
                        # We won't fallback to copy because user explicitly said "No copy"
                        # We won't fallback to hardlink because user explicitly said "No hardlink"
                        fail_count += 1
                        
            except Exception as e:
                console.print(f"[red]Error processing {src_path}: {e}[/red]")
                fail_count += 1
            
            progress.advance(task)
    
    console.print(f"\n[bold green]Export Complete![/bold green]")
    console.print(f"Processed: {len(image_destinations)} files")
    console.print(f"Operations: {success_count} successful, {fail_count} failed")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sorted photos.")
    parser.add_argument("output_dir", help="Directory to save sorted folders (FILES WILL BE MOVED HERE)")
    parser.add_argument("--unsorted", help="Optional directory to dump unsorted images")
    
    args = parser.parse_args()
    
    # Enforce Move+Link Logic
    export_photos(args.output_dir, mode='move', unsorted_dir=args.unsorted)
