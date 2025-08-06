# Pinterest Board Aesthetic Analysis

A comprehensive system for analyzing the visual evolution of your Pinterest boards using AI image embeddings and clustering.

## Features

🎨 **Visual Clustering**: Groups your Pinterest images by visual similarity using CLIP embeddings  
📊 **Aesthetic Evolution**: Tracks how your visual preferences change over time  
🏷️ **Automatic Labeling**: Generates descriptive labels for each visual theme cluster  
📈 **Growth Analysis**: Identifies growing, fading, and new aesthetic interests  

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision transformers scikit-learn hdbscan matplotlib pillow playwright requests tqdm numpy
playwright install chromium
```

### 2. Run Complete Analysis
```bash
# Full pipeline: scrape → cluster → analyze
python main.py --full https://pinterest.com/username/boardname

# Or step by step:
python main.py --scrape https://pinterest.com/username/boardname
python main.py --cluster
python main.py --analyze
```

### 3. Check Your Results
- **Cluster visualization**: `pinterest_images/clusters_plot.png`
- **Detailed assignments**: `pinterest_images/clusters.json`
- **Evolution report**: `pinterest_images/summary.md`

## Commands

| Command | Description |
|---------|-------------|
| `--scrape [URL]` | Download images from Pinterest board |
| `--cluster` | Analyze and group images by visual similarity |
| `--analyze` | Compare current clusters to previous analysis |
| `--full [URL]` | Complete pipeline in one command |
| `--status` | Show current collection statistics |

## How It Works

1. **Scraping**: Downloads images from your Pinterest board with metadata
2. **Embedding**: Converts images to 512-dimensional CLIP feature vectors
3. **Clustering**: Groups similar images using PCA + HDBSCAN/K-means
4. **Labeling**: Automatically generates descriptive names for each cluster
5. **Evolution**: Tracks changes between analysis runs

## Files Generated

```
pinterest_images/
├── *.jpg                    # Downloaded images
├── metadata.json            # Image URLs and metadata
├── clusters.json            # Cluster assignments with labels
├── clusters_plot.png        # Visual cluster plot
├── embeddings.npy           # CLIP embeddings for change analysis
├── summary.md               # Aesthetic evolution report
└── YYYY-MM-clusters.json    # Previous cluster backups
```

## Example Output

**Cluster Summary:**
- Cluster 0: Natural Elements (23 images)
- Cluster 1: Furniture Pieces (12 images)  
- Cluster 2: Outdoor Spaces (7 images)

**Evolution Report:**
- 🌱 **New Aesthetic Themes**: Minimalist Interiors (8 images)
- 📈 **Growing Interests**: Natural Elements (+45% growth)
- 🍂 **Diminishing Interests**: Vintage Furniture themes

## Advanced Usage

### Custom Analysis
```python
from cluster_images import embed_images, cluster_embeddings
from analyze_cluster_change import generate_summary

# Your custom analysis code here
```

### Batch Processing
```bash
# Analyze multiple boards
for url in $(cat board_urls.txt); do
    python main.py --full "$url"
done
```

## Technical Details

- **Image Embeddings**: OpenAI CLIP-ViT-B/32 (512 dimensions)
- **Dimensionality Reduction**: PCA with automatic component selection
- **Clustering**: HDBSCAN with K-means fallback for small datasets
- **Similarity Threshold**: 90% cosine similarity for cluster matching
- **Growth Threshold**: 20% increase to qualify as "growing"

## Troubleshooting

**"No clusters found"**: Your dataset might be too small or diverse. Try collecting more images.

**"Scraper found few images"**: Pinterest's infinite scroll may need longer wait times. Adjust `SCROLL_LIMIT` in `pinterest_scraper.py`.

**Import errors**: Ensure all dependencies are installed. Some require specific versions for compatibility.
