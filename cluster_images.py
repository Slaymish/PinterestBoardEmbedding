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
from sklearn.metrics import silhouette_score
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
        optimal_k = find_optimal_kmeans_clusters(reduced)
        print(f"Using {optimal_k} clusters for K-means")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
    
    return labels, reduced

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
    
    # Step 1: Try very broad categories first - comprehensive list covering diverse Pinterest content
    broad_concepts = [
        # Home & Decor
        "furniture and home decor", "interior design", "bedroom decor", "living room design", 
        "kitchen design", "bathroom decor", "outdoor and garden", "lighting and fixtures",
        "storage and organization", "wall art and decor", "minimalist design", "vintage decor",
        "modern contemporary design", "rustic farmhouse style", "boho bohemian decor",
        "scandinavian design", "industrial design", "cozy hygge style", "luxury home decor",
        
        # Food & Cooking
        "food and cooking", "baking and desserts", "healthy recipes", "comfort food",
        "meal prep", "cocktails and drinks", "food photography", "restaurant dishes",
        "breakfast ideas", "dinner recipes", "snacks and appetizers", "vegan food",
        "coffee and tea", "food styling", "kitchen gadgets",
        
        # Fashion & Beauty
        "fashion and style", "women's fashion", "men's fashion", "casual outfits",
        "formal wear", "shoes and accessories", "jewelry and watches", "bags and purses",
        "beauty and makeup", "hairstyles", "nail art", "skincare", "street style",
        "vintage fashion", "sustainable fashion", "plus size fashion", "wedding fashion",
        
        # Lifestyle & People
        "lifestyle photography", "people and portraits", "couples and relationships",
        "wedding inspiration", "pregnancy and maternity", "children and family",
        "cute babies", "pets and animals", "friendship goals", "date night ideas",
        "self care", "mental health", "fitness and wellness", "yoga and meditation",
        
        # Creative & Artistic
        "art and creativity", "paintings and artwork", "crafts and DIY", "handmade items",
        "creative projects", "artistic photography", "drawing and sketching", "pottery and ceramics",
        "woodworking", "sewing and embroidery", "paper crafts", "mixed media art",
        "street art", "digital art", "photography tips",
        
        # Specific Aesthetics & Trends
        "aesthetic photography", "dark academia", "cottagecore", "goblincore", "fairycore",
        "grunge aesthetic", "soft girl aesthetic", "y2k fashion", "indie aesthetic",
        "vintage aesthetic", "retro style", "pastel aesthetics", "monochrome style",
        "neon and bright colors", "earth tones", "neutral colors",
        
        # Architecture & Spaces
        "architecture and buildings", "tiny homes", "cabin retreats", "city apartments",
        "loft spaces", "greenhouse design", "outdoor living spaces", "pool areas",
        "treehouse design", "container homes", "architectural details", "room layouts",
        "space design", "commercial spaces", "retail design",
        
        # Nature & Outdoors
        "nature and plants", "houseplants", "garden design", "flowers and blooms",
        "landscape photography", "forest and trees", "mountains and hiking",
        "beach and ocean", "desert landscapes", "seasonal nature", "plant care",
        "botanical illustration", "succulents and cacti", "herb gardens",
        
        # Travel & Places
        "travel destinations", "vacation spots", "city views", "cultural sites",
        "road trip inspiration", "camping and outdoors", "hotel design", "restaurant design",
        "cafe aesthetics", "bookstore design", "museum interiors", "airport design",
        
        # Technology & Modern Life
        "technology and gadgets", "workspace design", "home office", "computer setup",
        "phone accessories", "smart home", "modern technology", "gaming setup",
        "productivity tools", "digital nomad", "tech aesthetics",
        
        # Textiles & Materials
        "textiles and fabrics", "fabric patterns", "textile art", "quilting",
        "knitting and crochet", "weaving", "clothing materials", "home textiles",
        "tapestries", "rugs and carpets", "curtains and drapes",
        
        # Specific Items & Objects
        "books and reading", "stationery and supplies", "candles and lighting",
        "mirrors and reflections", "clocks and time", "musical instruments",
        "sporting goods", "tools and hardware", "collectibles", "vintage items",
        
        # Seasonal & Holiday
        "christmas decorations", "halloween decor", "spring aesthetics", "summer vibes",
        "autumn colors", "winter cozy", "holiday themes", "party decorations",
        "birthday celebrations", "seasonal crafts",
        
        # Unique & Niche Concepts
        "hammock indoors", "reading nooks", "cozy corners", "study spaces",
        "creative workspaces", "meditation spaces", "exercise areas", "hobby rooms",
        "cute aesthetic", "soft aesthetics", "dark moody", "bright and airy",
        "maximalist design", "eclectic style", "punk aesthetic", "gothic style"
    ]
    
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
    """Generate more specific concepts based on a broad category"""
    concept_refinements = {
        "furniture and home decor": [
            "chairs and seating", "tables and surfaces", "storage furniture", "decorative objects",
            "wall art and decor", "shelving and organization", "lighting fixtures", "textiles and cushions",
            "vintage furniture", "modern furniture", "rustic furniture", "luxury furniture"
        ],
        "food and cooking": [
            "prepared meals", "baking and desserts", "fresh ingredients", "cooking utensils",
            "kitchen appliances", "food presentation", "beverages and drinks", "healthy food",
            "comfort food", "gourmet cuisine", "street food", "homemade cooking"
        ],
        "fashion and style": [
            "clothing and outfits", "shoes and accessories", "jewelry and watches", "bags and purses",
            "beauty and makeup", "hairstyles", "fashion trends", "style inspiration",
            "casual fashion", "formal wear", "street style", "vintage fashion", "alternative fashion"
        ],
        "art and creativity": [
            "paintings and artwork", "crafts and DIY", "creative projects", "artistic supplies",
            "handmade items", "creative workspace", "artistic techniques", "mixed media art",
            "digital art", "street art", "abstract art", "portrait art"
        ],
        "nature and plants": [
            "houseplants and greenery", "flowers and blooms", "garden design", "outdoor plants",
            "plant care", "botanical photography", "natural textures", "seasonal plants",
            "succulents", "tropical plants", "herb gardens", "flower arrangements"
        ],
        "architecture and buildings": [
            "interior design", "exterior architecture", "room layouts", "architectural details",
            "building facades", "construction and renovation", "structural elements", "space design",
            "tiny homes", "modern architecture", "historic buildings", "commercial spaces"
        ],
        "technology and gadgets": [
            "electronic devices", "computer equipment", "smart home technology", "mobile devices",
            "audio equipment", "tech accessories", "digital displays", "modern technology",
            "gaming technology", "workspace tech", "wearable technology"
        ],
        "textiles and fabrics": [
            "fabric patterns", "textile textures", "clothing materials", "home textiles",
            "embroidery and stitching", "fabric colors", "woven materials", "soft furnishings",
            "quilting", "knitting", "crochet work", "fabric art"
        ],
        "lifestyle photography": [
            "candid moments", "daily life", "personal style", "lifestyle inspiration",
            "cozy moments", "aesthetic photography", "mood photography", "storytelling photos"
        ],
        "people and portraits": [
            "portrait photography", "candid people", "group photos", "family portraits",
            "professional headshots", "artistic portraits", "lifestyle portraits", "cute people"
        ],
        "couples and relationships": [
            "couple goals", "romantic moments", "date ideas", "relationship inspiration",
            "engagement photos", "cute couples", "love aesthetic", "anniversary ideas"
        ],
        "hairstyles": [
            "short haircuts", "long hairstyles", "curly hair", "straight hair", "braided styles",
            "hair color ideas", "trendy cuts", "classic styles", "wolf haircuts", "layered cuts",
            "pixie cuts", "bob haircuts", "hair accessories", "styling techniques"
        ],
        "aesthetic photography": [
            "soft aesthetic", "dark academia", "cottagecore", "grunge aesthetic", "vintage aesthetic",
            "minimalist aesthetic", "maximalist aesthetic", "indie aesthetic", "y2k aesthetic",
            "fairycore", "goblincore", "light academia", "dark feminine"
        ],
        "cozy hygge style": [
            "cozy interiors", "hygge lifestyle", "warm lighting", "comfortable spaces",
            "reading nooks", "cozy corners", "soft textures", "comfort items",
            "hammock indoors", "cozy bedrooms", "fireplace areas", "warm aesthetics"
        ],
        "cute aesthetic": [
            "kawaii style", "pastel colors", "soft aesthetics", "adorable items",
            "cute decor", "sweet treats", "plushies", "cute animals", "soft girl aesthetic"
        ]
    }
    
    return concept_refinements.get(broad_concept, [broad_concept])

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
