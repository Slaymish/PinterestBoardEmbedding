# analyze_cluster_change.py

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import date
import shutil
import glob
from typing import Dict, List, Optional

IMG_DIR = Path("pinterest_images")
CURRENT_FILE = IMG_DIR / "clusters.json"
EMBEDDINGS_FILE = IMG_DIR / "embeddings.npy"
SUMMARY_FILE = IMG_DIR / "summary.md"

def find_previous_cluster_file() -> Optional[Path]:
    """Find the most recent previous cluster backup file"""
    backup_files = glob.glob(str(IMG_DIR / "*-*-clusters.json"))
    if not backup_files:
        print("No previous cluster files found")
        return None
    
    # Sort by date (filename format: YYYY-MM-clusters.json)
    backup_files.sort(reverse=True)
    prev_file = Path(backup_files[0])
    print(f"Using previous cluster file: {prev_file.name}")
    return prev_file

def load_clusters(path: Path) -> Dict:
    """Load cluster data from JSON file"""
    if not path.exists():
        print(f"Cluster file {path} not found")
        return {}
    
    with open(path) as f:
        data = json.load(f)
    
    clusters = defaultdict(list)
    cluster_labels = {}
    
    for url, entry in data.items():
        cluster_id = entry["cluster"]
        clusters[cluster_id].append(entry["filename"])
        
        # Store cluster labels if available
        if "cluster_label" in entry:
            cluster_labels[cluster_id] = entry["cluster_label"]
    
    return {"clusters": clusters, "labels": cluster_labels}

def compute_centroids(clusters, embeddings):
    centroids = {}
    for label, filenames in clusters.items():
        vecs = [embeddings[f] for f in filenames if f in embeddings]
        if vecs:
            centroids[label] = np.mean(vecs, axis=0)
    return centroids

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def match_clusters(current_centroids, previous_centroids, threshold=0.9):
    matches = {}
    for curr_label, curr_vec in current_centroids.items():
        best_score = -1
        best_match = None
        for prev_label, prev_vec in previous_centroids.items():
            score = cosine_similarity(curr_vec, prev_vec)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = prev_label
        matches[curr_label] = best_match
    return matches

def load_embeddings() -> Dict[str, np.ndarray]:
    """Load embeddings from the most recent clustering run"""
    if not EMBEDDINGS_FILE.exists():
        print(f"Embeddings file {EMBEDDINGS_FILE} not found")
        print("Run cluster_images.py first to generate embeddings")
        return {}
    
    try:
        embeddings_data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        print(f"Loaded embeddings for {len(embeddings_data)} images")
        return embeddings_data
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}

def generate_summary(matches: Dict, curr_data: Dict, prev_data: Dict):
    """Generate a markdown summary of cluster changes"""
    curr_clusters = curr_data["clusters"]
    prev_clusters = prev_data["clusters"] 
    curr_labels = curr_data["labels"]
    prev_labels = prev_data["labels"]
    
    new, fading, growing, stable = [], [], [], []

    reverse_matches = {v: k for k, v in matches.items() if v is not None}

    for curr, prev in matches.items():
        if prev is None:
            new.append(curr)
        else:
            curr_size = len(curr_clusters[curr])
            prev_size = len(prev_clusters[prev])
            
            if curr_size > prev_size * 1.2:  # 20% growth threshold
                growing.append(curr)
            elif abs(curr_size - prev_size) <= max(1, prev_size * 0.1):  # Within 10%
                stable.append(curr)

    for prev in prev_clusters:
        if prev not in reverse_matches:
            fading.append(prev)

    with open(SUMMARY_FILE, "w") as f:
        f.write(f"# Pinterest Aesthetic Evolution Report\n")
        f.write(f"**Generated:** {date.today().strftime('%Y-%m-%d')}\n\n")
        
        f.write(f"**Current Collection:** {sum(len(imgs) for imgs in curr_clusters.values())} images across {len(curr_clusters)} clusters\n\n")

        if new:
            f.write("## 🌱 New Aesthetic Themes\n")
            for c in new:
                label = curr_labels.get(c, f"Cluster {c}")
                f.write(f"- **{label}**: {len(curr_clusters[c])} images\n")
            f.write("\n")

        if growing:
            f.write("## 📈 Growing Interests\n")
            for c in growing:
                label = curr_labels.get(c, f"Cluster {c}")
                prev_match = matches[c]
                prev_size = len(prev_clusters[prev_match]) if prev_match else 0
                curr_size = len(curr_clusters[c])
                growth = ((curr_size - prev_size) / prev_size * 100) if prev_size > 0 else 100
                f.write(f"- **{label}**: {curr_size} images (+{growth:.1f}% growth)\n")
            f.write("\n")

        if stable:
            f.write("## 🎯 Consistent Themes\n")
            for c in stable:
                label = curr_labels.get(c, f"Cluster {c}")
                f.write(f"- **{label}**: {len(curr_clusters[c])} images\n")
            f.write("\n")

        if fading:
            f.write("## 🍂 Diminishing Interests\n")
            for c in fading:
                label = prev_labels.get(c, f"Cluster {c}")
                f.write(f"- **{label}**: No longer prominent in current collection\n")
            f.write("\n")

        if not (new or growing or fading):
            f.write("## 😌 Stable Aesthetic\n")
            f.write("Your visual interests have remained consistent since the last analysis.\n\n")

    print(f"✅ Evolution report saved to {SUMMARY_FILE}")
    print(f"   📊 {len(new)} new themes, {len(growing)} growing, {len(fading)} fading")

def backup_current_file():
    today = date.today()
    backup_name = f"{today.year}-{today.month:02d}-clusters.json"
    shutil.copy(CURRENT_FILE, IMG_DIR / backup_name)

if __name__ == "__main__":
    print("🔍 Analyzing Pinterest aesthetic evolution...")
    
    # Check if current clusters exist
    if not CURRENT_FILE.exists():
        print("❌ No current clusters.json found")
        print("   Run cluster_images.py first to generate clusters")
        exit(1)
    
    # Find previous cluster file
    previous_file = find_previous_cluster_file()
    if not previous_file:
        print("⚠️  No previous cluster data found - creating initial backup")
        backup_current_file()
        print("   Run this script again after your next clustering to see changes")
        exit(0)
    
    # Load data
    curr_data = load_clusters(CURRENT_FILE)
    prev_data = load_clusters(previous_file)
    embeddings = load_embeddings()
    
    if not embeddings:
        print("❌ Could not load embeddings")
        exit(1)
    
    # Compute centroids and match clusters
    curr_centroids = compute_centroids(curr_data["clusters"], embeddings)
    prev_centroids = compute_centroids(prev_data["clusters"], embeddings)
    
    matches = match_clusters(curr_centroids, prev_centroids)
    
    # Generate summary and backup
    generate_summary(matches, curr_data, prev_data)
    backup_current_file()
    
    print("✅ Analysis complete!")
