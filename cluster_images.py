# cluster_images.py

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan

from transformers import CLIPProcessor, CLIPModel

IMG_DIR = Path("pinterest_images")
METADATA_FILE = IMG_DIR / "metadata.json"
CLUSTER_FILE = IMG_DIR / "clusters.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_metadata():
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def load_images(metadata, max_images=None):
    images = []
    urls = []
    filenames = []
    for url, meta in list(metadata.items())[:max_images]:
        path = IMG_DIR / meta["filename"]
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                urls.append(url)
                filenames.append(meta["filename"])
            except:
                continue
    return images, urls, filenames

def embed_images(images):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    embeddings = []

    for i in tqdm(range(0, len(images), 8), desc="Embedding"):
        batch = images[i:i+8]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embeddings.append(outputs.cpu().numpy())

    return np.vstack(embeddings)

def cluster_embeddings(embeddings):
    # Test different component amounts to find optimal variance explained
    max_components = min(50, embeddings.shape[0] - 1)
    test_components = [2, 5, 10, 15, 20, max_components]
    # Remove duplicates and keep only valid options
    test_components = sorted(list(set([c for c in test_components if c >= 2 and c <= max_components])))
    
    best_components = 2
    best_variance_ratio = 0
    
    print(f"Testing PCA components for optimal variance explained...")
    for n_comp in test_components:
        reducer = PCA(n_components=n_comp)
        reducer.fit(embeddings)
        total_variance = np.sum(reducer.explained_variance_ratio_)
        print(f"  {n_comp} components: {total_variance:.3f} variance explained")
        
        if total_variance > best_variance_ratio:
            best_variance_ratio = total_variance
            best_components = n_comp
    
    print(f"Selected {best_components} components (explains {best_variance_ratio:.3f} of variance)")
    
    # Use the best number of components
    reducer = PCA(n_components=best_components)
    reduced = reducer.fit_transform(embeddings)

    # For small datasets, use very relaxed clustering parameters
    min_cluster_size = max(2, embeddings.shape[0] // 6)  # More aggressive clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,  # Very permissive
        cluster_selection_epsilon=0.1,  # Allow looser clusters
        alpha=1.0  # More aggressive cluster formation
    )
    labels = clusterer.fit_predict(reduced)
    
    # If still no clusters found, try K-means as fallback
    if len(set(labels)) <= 1 or all(label == -1 for label in labels):
        print("HDBSCAN found no clusters, trying K-means as fallback...")
        n_clusters = min(3, max(2, embeddings.shape[0] // 4))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
    
    return labels, reduced

def plot_clusters(reduced, labels, images, filenames):
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f"Cluster {label}" if label != -1 else "Noise", s=15)

    plt.legend()
    plt.title("Pinterest Image Clusters")
    plt.savefig(IMG_DIR / "clusters_plot.png")
    plt.close()

def generate_cluster_labels(images, labels, urls, filenames):
    """Generate descriptive labels for each cluster using CLIP text-image similarity"""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    
    unique_labels = set(labels)
    cluster_descriptions = {}
    
    print("Generating cluster labels...")
    
    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise
            cluster_descriptions[cluster_id] = "Mixed/Uncategorized"
            continue
            
        # Get images in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_images = [images[i] for i in cluster_indices]
        
        if not cluster_images:
            cluster_descriptions[cluster_id] = "Empty Cluster"
            continue
        
        # Sample up to 3 images from the cluster for efficiency
        sample_images = cluster_images[:3]
        
        # Use a more diverse set of potential descriptions
        # Generate descriptions dynamically based on common visual elements
        broad_categories = [
            "architectural elements", "interior spaces", "furniture pieces",
            "decorative objects", "textual content", "natural elements",
            "food items", "fashion accessories", "artistic compositions",
            "people and portraits", "vehicles and transportation", "technology items",
            "patterns and textures", "lighting fixtures", "storage solutions",
            "kitchen appliances", "bathroom fixtures", "outdoor spaces",
            "plants and flowers", "color schemes", "geometric shapes"
        ]
        
        # Calculate similarity scores for each broad category
        best_score = -1
        best_label = "Unknown"
        
        print(f"  Analyzing Cluster {cluster_id}...")
        
        for category in tqdm(broad_categories, desc=f"Testing categories", leave=False):
            scores = []
            
            for img in sample_images:
                # Test multiple phrasings for each category
                test_phrases = [
                    f"a photo of {category}",
                    f"an image showing {category}",
                    f"{category} in home design"
                ]
                
                phrase_scores = []
                for phrase in test_phrases:
                    inputs = processor(
                        text=[phrase], 
                        images=[img], 
                        return_tensors="pt", 
                        padding=True
                    ).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        similarity = outputs.logits_per_image[0][0].item()
                        phrase_scores.append(similarity)
                
                scores.append(max(phrase_scores))  # Take best phrase score
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_label = category
        
        # Clean up the label
        cluster_descriptions[cluster_id] = best_label.replace("_", " ").title()
        print(f"  Cluster {cluster_id}: {cluster_descriptions[cluster_id]} (score: {best_score:.2f})")
    
    return cluster_descriptions

def save_cluster_data(urls, filenames, labels, cluster_descriptions=None):
    clusters = {}
    for url, filename, label in zip(urls, filenames, labels):
        cluster_info = {
            "filename": filename,
            "cluster": int(label),
        }
        
        # Add cluster description if available
        if cluster_descriptions and label in cluster_descriptions:
            cluster_info["cluster_label"] = cluster_descriptions[label]
            
        clusters[url] = cluster_info
        
    with open(CLUSTER_FILE, "w") as f:
        json.dump(clusters, f, indent=2)

if __name__ == "__main__":
    metadata = load_metadata()
    images, urls, filenames = load_images(metadata)
    embeddings = embed_images(images)
    
    # Save embeddings for change analysis
    embeddings_dict = {filename: embedding for filename, embedding in zip(filenames, embeddings)}
    np.save(IMG_DIR / "embeddings.npy", embeddings_dict)
    print(f"Saved embeddings for {len(embeddings_dict)} images")
    
    labels, reduced = cluster_embeddings(embeddings)
    plot_clusters(reduced, labels, images, filenames)
    
    # Generate descriptive labels for clusters
    cluster_descriptions = generate_cluster_labels(images, labels, urls, filenames)
    
    save_cluster_data(urls, filenames, labels, cluster_descriptions)
    print(f"Saved cluster plot and assignments to {CLUSTER_FILE}")
    
    # Print summary
    print("\nCluster Summary:")
    for cluster_id, description in cluster_descriptions.items():
        count = sum(1 for label in labels if label == cluster_id)
        print(f"  Cluster {cluster_id}: {description} ({count} images)")
