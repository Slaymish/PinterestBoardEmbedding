#!/usr/bin/env python3
"""
Pinterest Board Embedding Analysis - Main Script

A cohesive system for scraping Pinterest boards, clustering images by visual similarity,
and tracking how your aesthetic preferences evolve over time.

Usage:
    python main.py --scrape [URL]     # Scrape new images from Pinterest board
    python main.py --cluster          # Cluster existing images
    python main.py --analyze          # Analyze changes from previous clustering
    python main.py --full [URL]       # Complete workflow: scrape + cluster + analyze
    python main.py --status           # Show current collection status
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

IMG_DIR = Path("pinterest_images")
METADATA_FILE = IMG_DIR / "metadata.json"
CLUSTERS_FILE = IMG_DIR / "clusters.json"

# Default Pinterest board
DEFAULT_BOARD_URL = "https://nz.pinterest.com/slaymish/in-the-room/"

def get_collection_stats():
    """Get current collection statistics"""
    stats = {"images": 0, "clusters": 0, "has_labels": False}
    
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        stats["images"] = len(metadata)
    
    if CLUSTERS_FILE.exists():
        with open(CLUSTERS_FILE) as f:
            clusters = json.load(f)
        unique_clusters = set(entry["cluster"] for entry in clusters.values())
        stats["clusters"] = len(unique_clusters)
        stats["has_labels"] = any("cluster_label" in entry for entry in clusters.values())
    
    return stats

def run_scraper(board_url=None):
    """Run the Pinterest scraper"""
    print("🔍 Starting Pinterest scraper...")
    
    if board_url and board_url != "interactive":
        # Run with provided URL
        process = subprocess.run([sys.executable, "pinterest_scraper.py"], 
                               input=board_url, text=True, capture_output=True)
    elif board_url == "interactive":
        # Interactive mode
        process = subprocess.run([sys.executable, "pinterest_scraper.py"])
    else:
        # Use default board - send empty line to trigger default
        print(f"Using default board: {DEFAULT_BOARD_URL}")
        process = subprocess.run([sys.executable, "pinterest_scraper.py"], 
                               input="\n", text=True, capture_output=True)
    
    if process.returncode != 0:
        print(f"❌ Scraper failed: {process.stderr}")
        return False
    
    # Show scraper output for feedback
    if process.stdout:
        print(process.stdout)
    
    print("✅ Scraping completed")
    return True

def run_clustering():
    """Run image clustering"""
    print("🎨 Starting image clustering...")
    
    process = subprocess.run([sys.executable, "cluster_images.py"])
    
    if process.returncode != 0:
        print("❌ Clustering failed")
        return False
    
    print("✅ Clustering completed")
    return True

def run_analysis():
    """Run aesthetic change analysis"""
    print("📊 Starting aesthetic evolution analysis...")
    
    process = subprocess.run([sys.executable, "analyze_cluster_change.py"])
    
    if process.returncode != 0:
        print("❌ Analysis failed")
        return False
    
    print("✅ Analysis completed")
    return True

def show_status():
    """Show current collection status"""
    stats = get_collection_stats()
    
    print("📊 Current Collection Status")
    print("=" * 40)
    print(f"Images collected: {stats['images']}")
    print(f"Clusters identified: {stats['clusters']}")
    print(f"Has cluster labels: {'Yes' if stats['has_labels'] else 'No'}")
    
    if stats["images"] == 0:
        print(f"\n💡 Get started: python main.py --scrape")
        print(f"   (Will use default board: {DEFAULT_BOARD_URL})")
    elif stats["clusters"] == 0:
        print("\n💡 Next step: python main.py --cluster")
    else:
        print("\n💡 Track changes: python main.py --analyze")

def main():
    parser = argparse.ArgumentParser(
        description="Pinterest Board Aesthetic Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--scrape", metavar="URL", nargs="?", const="default",
                       help=f"Scrape Pinterest board (default: your 'In The Room' board)")
    parser.add_argument("--cluster", action="store_true",
                       help="Cluster images by visual similarity")
    parser.add_argument("--analyze", action="store_true", 
                       help="Analyze aesthetic changes over time")
    parser.add_argument("--full", metavar="URL", nargs="?", const="default",
                       help=f"Run complete pipeline: scrape + cluster + analyze (default board)")
    parser.add_argument("--status", action="store_true",
                       help="Show current collection status")
    
    args = parser.parse_args()
    
    # Ensure pinterest_images directory exists
    IMG_DIR.mkdir(exist_ok=True)
    
    if args.status:
        show_status()
    
    elif args.scrape:
        if args.scrape == "default":
            url = None  # Will use default
        elif args.scrape == "interactive":
            url = "interactive"
        else:
            url = args.scrape
            
        success = run_scraper(url)
        if success:
            show_status()
    
    elif args.cluster:
        stats = get_collection_stats()
        if stats["images"] == 0:
            print("❌ No images found. Run scraper first:")
            print("   python main.py --scrape")
            return
        
        success = run_clustering()
        if success:
            show_status()
    
    elif args.analyze:
        if not CLUSTERS_FILE.exists():
            print("❌ No clusters found. Run clustering first:")
            print("   python main.py --cluster")
            return
        
        run_analysis()
    
    elif args.full:
        if args.full == "default":
            url = None  # Will use default
        elif args.full == "interactive":
            url = "interactive"  
        else:
            url = args.full
        
        print("🚀 Running complete Pinterest analysis pipeline...")
        
        # Step 1: Scrape
        if not run_scraper(url):
            return
        
        # Step 2: Cluster
        if not run_clustering():
            return
        
        # Step 3: Analyze
        run_analysis()
        
        print("🎉 Complete pipeline finished!")
        show_status()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
