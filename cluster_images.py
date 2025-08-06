# cluster_images.py

import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

from transformers import CLIPProcessor, CLIPModel

IMG_DIR = Path("pinterest_images")
METADATA_FILE = IMG_DIR / "metadata.json"
CLUSTER_FILE = IMG_DIR / "clusters.json"
CATEGORIES_FILE = Path("cluster_categories.csv")
REFINEMENTS_FILE = Path("cluster_refinements.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_broad_categories():
    """Load broad categories from CSV file"""
    categories = []
    try:
        with open(CATEGORIES_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                categories.append(row['category_name'])
    except FileNotFoundError:
        print(f"Warning: {CATEGORIES_FILE} not found, using default categories")
        # Fallback to a basic set if file doesn't exist
        categories = [
            "furniture and home decor", "food and cooking", "fashion and style", 
            "art and creativity", "nature and plants", "architecture and buildings",
            "technology and gadgets", "aesthetic photography"
        ]
    return categories

def load_refinement_mapping():
    """Load category refinements mapping from CSV file"""
    refinements = {}
    try:
        with open(REFINEMENTS_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                broad_cat = row['broad_category']
                refined_cat = row['refined_category']
                
                if broad_cat not in refinements:
                    refinements[broad_cat] = []
                refinements[broad_cat].append(refined_cat)
    except FileNotFoundError:
        print(f"Warning: {REFINEMENTS_FILE} not found, using default refinements")
        # Return empty dict, will fallback to broad category
        refinements = {}
    
    return refinements

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

    # Try HDBSCAN first with optimized parameters
    labels = try_optimized_hdbscan(reduced)
    
    # If HDBSCAN didn't work well, try optimized K-means
    if len(set(labels)) <= 1 or all(label == -1 for label in labels) or len(set(labels)) > embeddings.shape[0] // 3:
        print("HDBSCAN results not optimal, trying K-means with optimization...")
        optimal_k = find_optimal_kmeans_clusters(reduced)
        print(f"Using {optimal_k} clusters for K-means")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
    
    return labels, reduced

def try_optimized_hdbscan(data):
    """Try HDBSCAN with multiple parameter combinations to find best clustering"""
    n_samples = data.shape[0]
    
    # Test different HDBSCAN parameters
    parameter_combinations = [
        # (min_cluster_size, min_samples, cluster_selection_epsilon)
        (max(2, n_samples // 8), 1, 0.1),     # Very permissive
        (max(3, n_samples // 6), 2, 0.15),    # Moderately permissive  
        (max(4, n_samples // 5), 3, 0.2),     # More conservative
        (max(5, n_samples // 4), 2, 0.1),     # Alternative approach
    ]
    
    best_labels = None
    best_score = -2  # Worse than worst silhouette score
    best_params = None
    
    print("Testing HDBSCAN parameter combinations...")
    
    for min_cluster_size, min_samples, epsilon in parameter_combinations:
        if min_cluster_size >= n_samples // 2:  # Skip if too restrictive
            continue
            
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            alpha=1.0
        )
        
        labels = clusterer.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
        print(f"    → {n_clusters} clusters, {n_noise} noise points")
        
        # Score this clustering attempt
        if n_clusters >= 2:  # Need at least 2 clusters to calculate silhouette
            try:
                # Filter out noise points for silhouette calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:  # Need more than 1 non-noise point
                    filtered_data = data[non_noise_mask]
                    filtered_labels = labels[non_noise_mask]
                    
                    if len(set(filtered_labels)) > 1:  # Multiple clusters
                        score = silhouette_score(filtered_data, filtered_labels)
                        
                        # Penalty for too much noise or too many tiny clusters
                        noise_penalty = n_noise / n_samples * 0.5
                        cluster_size_penalty = 0
                        if n_clusters > n_samples // 4:  # Too many small clusters
                            cluster_size_penalty = 0.3
                        
                        adjusted_score = score - noise_penalty - cluster_size_penalty
                        
                        print(f"    → silhouette: {score:.3f}, adjusted: {adjusted_score:.3f}")
                        
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_labels = labels.copy()
                            best_params = (min_cluster_size, min_samples, epsilon)
                    else:
                        print(f"    → only one cluster after noise removal")
                else:
                    print(f"    → too many noise points")
            except Exception as e:
                print(f"    → error calculating silhouette: {e}")
        else:
            print(f"    → insufficient clusters")
    
    if best_labels is not None:
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = list(best_labels).count(-1)
        print(f"Selected HDBSCAN: {n_clusters} clusters, {n_noise} noise (score: {best_score:.3f})")
        print(f"Parameters: min_cluster_size={best_params[0]}, min_samples={best_params[1]}, epsilon={best_params[2]}")
        return best_labels
    else:
        print("No good HDBSCAN clustering found")
        return np.array([-1] * len(data))  # All noise

def find_optimal_kmeans_clusters(data):
    """Find optimal number of clusters using elbow method and silhouette score"""
    n_samples = data.shape[0]
    
    # Test range of clusters - from 2 to reasonable maximum
    max_k = min(8, max(2, n_samples // 3))  # Don't go crazy with cluster count
    k_range = range(2, max_k + 1)
    
    inertias = []
    silhouette_scores = []
    
    print("Finding optimal K-means clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score (higher is better)
        sil_score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(sil_score)
        
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.3f}")
    
    # Find elbow point using rate of change in inertia
    if len(inertias) >= 3:
        # Calculate second derivative to find elbow
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # Find the k with maximum second derivative (sharpest elbow)
        elbow_idx = second_derivatives.index(max(second_derivatives))
        elbow_k = k_range[elbow_idx + 1]  # +1 because second_derivatives is offset
    else:
        elbow_k = k_range[0]  # fallback to minimum k
    
    # Find k with best silhouette score
    best_silhouette_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    
    # Decision logic: prefer silhouette if it's significantly better, otherwise use elbow
    best_sil_score = max(silhouette_scores)
    elbow_sil_score = silhouette_scores[elbow_k - k_range[0]]
    
    if best_sil_score > elbow_sil_score + 0.1:  # Silhouette is significantly better
        optimal_k = best_silhouette_k
        print(f"  → Selected k={optimal_k} (best silhouette score: {best_sil_score:.3f})")
    else:
        optimal_k = elbow_k
        print(f"  → Selected k={optimal_k} (elbow method, silhouette: {elbow_sil_score:.3f})")
    
    return optimal_k

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

def generate_dynamic_cluster_labels(images, labels, urls, filenames):
    """Generate descriptive labels for each cluster using CLIP with dynamic concept discovery"""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    
    unique_labels = set(labels)
    cluster_descriptions = {}
    
    print("Generating truly dynamic cluster labels...")
    
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
        
        print(f"  Analyzing Cluster {cluster_id} ({len(cluster_images)} images)...")
        
        # Sample up to 3 images from the cluster for efficiency
        sample_size = min(3, len(cluster_images))
        sample_images = cluster_images[:sample_size]
        
        # Use a hierarchical approach to discover what the cluster represents
        cluster_label = discover_cluster_concept(sample_images, processor, model)
        cluster_descriptions[cluster_id] = cluster_label
        print(f"  Cluster {cluster_id}: {cluster_label}")
    
    return cluster_descriptions

def discover_cluster_concept(images, processor, model):
    """Dynamically discover what concept best describes a set of images"""
    
    # Step 1: Load broad categories from CSV file
    broad_concepts = load_broad_categories()
    
    best_broad_score = -1
    best_broad_concept = None
    
    # Find the best broad category
    for concept in broad_concepts:
        scores = []
        for img in images:
            inputs = processor(
                text=[f"a photo of {concept}"], 
                images=[img], 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                similarity = outputs.logits_per_image[0][0].item()
                scores.append(similarity)
        
        avg_score = np.mean(scores)
        if avg_score > best_broad_score:
            best_broad_score = avg_score
            best_broad_concept = concept
    
    # Step 2: Refine the concept based on the best broad category
    refined_concepts = generate_refined_concepts(best_broad_concept)
    
    best_refined_score = best_broad_score
    best_concept = best_broad_concept
    
    # Test refined concepts
    for concept in refined_concepts:
        scores = []
        for img in images:
            inputs = processor(
                text=[f"a photo of {concept}"], 
                images=[img], 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                similarity = outputs.logits_per_image[0][0].item()
                scores.append(similarity)
        
        avg_score = np.mean(scores)
        if avg_score > best_refined_score:
            best_refined_score = avg_score
            best_concept = concept
    
    # Step 3: Try some specific descriptive variations
    final_concepts = generate_specific_variations(best_concept)
    
    best_final_score = best_refined_score
    final_concept = best_concept
    
    for concept in final_concepts:
        scores = []
        for img in images:
            inputs = processor(
                text=[concept], 
                images=[img], 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                similarity = outputs.logits_per_image[0][0].item()
                scores.append(similarity)
        
        avg_score = np.mean(scores)
        if avg_score > best_final_score:
            best_final_score = avg_score
            final_concept = concept
    
    # Clean up and return the best concept
    return clean_concept_label(final_concept)

def generate_refined_concepts(broad_concept):
    """Generate more specific concepts based on a broad category from CSV data"""
    refinement_mapping = load_refinement_mapping()
    
    # Get refinements from CSV, fallback to the concept itself if not found
    refined_concepts = refinement_mapping.get(broad_concept, [broad_concept])
    
    return refined_concepts

def generate_specific_variations(concept):
    """Generate specific descriptive variations of a concept"""
    # More diverse adjectives and style descriptors
    style_variations = [
        concept,
        f"cute {concept}",
        f"aesthetic {concept}",
        f"trendy {concept}",
        f"modern {concept}",
        f"stylish {concept}",
        f"beautiful {concept}",
        f"elegant {concept}",
        f"cozy {concept}",
        f"minimalist {concept}",
        f"vintage {concept}",
        f"retro {concept}",
        f"colorful {concept}",
        f"pastel {concept}",
        f"dark {concept}",
        f"soft {concept}",
        f"grunge {concept}",
        f"indie {concept}",
        f"alternative {concept}",
        f"unique {concept}",
        f"creative {concept}",
        f"artistic {concept}",
        f"luxury {concept}",
        f"rustic {concept}",
        f"contemporary {concept}",
        f"bohemian {concept}",
        f"scandinavian {concept}",
        f"industrial {concept}",
        f"eclectic {concept}",
        f"maximalist {concept}",
        f"cottagecore {concept}",
        f"dark academia {concept}",
        f"y2k {concept}",
        f"fairycore {concept}",
        f"kawaii {concept}"
    ]
    
    # Alternative phrasings and contexts
    context_variations = [
        concept,
        f"image of {concept}",
        f"picture showing {concept}",
        f"photo featuring {concept}",
        f"{concept} photography",
        f"{concept} aesthetic",
        f"{concept} inspiration",
        f"{concept} ideas",
        f"{concept} design",
        f"{concept} style",
        f"{concept} vibes",
        f"{concept} mood",
        f"{concept} goals",
        f"{concept} trends",
        f"pinterest {concept}",
        f"tumblr {concept}",
        f"instagram {concept}",
        f"{concept} content",
        f"{concept} posts",
        f"{concept} board"
    ]
    
    # Combine both types of variations
    all_variations = style_variations + context_variations
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for item in all_variations:
        if item not in seen:
            seen.add(item)
            unique_variations.append(item)
    
    return unique_variations

def clean_concept_label(concept):
    """Clean up the concept label for final presentation"""
    # Remove common prefixes
    prefixes_to_remove = [
        "a photo of ", "an image of ", "image of ", "picture showing ",
        "photography", "design", "inspiration"
    ]
    
    cleaned = concept
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove "and" connectors for cleaner labels
    if " and " in cleaned:
        parts = cleaned.split(" and ")
        # Take the first part if it's substantial, otherwise keep both
        if len(parts[0].split()) >= 2:
            cleaned = parts[0]
    
    # Capitalize properly
    return cleaned.title()

def generate_cluster_labels(images, labels, urls, filenames):
    """Generate descriptive labels for each cluster - now using dynamic approach"""
    try:
        return generate_dynamic_cluster_labels(images, labels, urls, filenames)
    except Exception as e:
        print(f"Dynamic labeling failed ({e}), falling back to basic CLIP approach...")
        return generate_basic_clip_labels(images, labels, urls, filenames)

def generate_basic_clip_labels(images, labels, urls, filenames):
    """Simple fallback method using basic CLIP categories"""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    
    unique_labels = set(labels)
    cluster_descriptions = {}
    
    print("Using basic CLIP fallback for cluster labeling...")
    
    # Simple, broad categories for fallback
    basic_categories = [
        "furniture", "food", "fashion", "plants", "decor", "art", "kitchen items", 
        "bedroom items", "bathroom items", "outdoor items", "technology", "textiles"
    ]
    
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
        
        # Sample images for analysis
        sample_images = cluster_images[:2]
        
        best_score = -1
        best_label = "Unknown"
        
        for category in basic_categories:
            scores = []
            for img in sample_images:
                inputs = processor(
                    text=[f"a photo of {category}"], 
                    images=[img], 
                    return_tensors="pt", 
                    padding=True
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    similarity = outputs.logits_per_image[0][0].item()
                    scores.append(similarity)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_label = category
        
        cluster_descriptions[cluster_id] = best_label.title()
    
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
