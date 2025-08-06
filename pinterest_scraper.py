import os
import json
import time
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from hashlib import sha256
from pathlib import Path
from playwright.sync_api import sync_playwright

SAVE_DIR = Path("pinterest_images")
METADATA_FILE = SAVE_DIR / "metadata.json"
SCROLL_LIMIT = 3000  # pixels to scroll down

# Default board to scrape
DEFAULT_BOARD_URL = "https://nz.pinterest.com/slaymish/in-the-room/"

def ensure_dir():
    SAVE_DIR.mkdir(exist_ok=True)
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w") as f:
            json.dump({}, f)

def generate_filename(url: str) -> str:
    return sha256(url.encode()).hexdigest() + ".jpg"

def download_image(url: str, filename: str) -> bool:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and 'image' in r.headers.get("content-type", ""):
            with open(SAVE_DIR / filename, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def scrape_images(board_url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Set a larger viewport to load more images
        page.set_viewport_size({"width": 1920, "height": 1080})
        
        print(f"Opening {board_url}")
        page.goto(board_url, wait_until="networkidle")
        
        # Wait for initial content to load
        time.sleep(3)
        
        # More aggressive scrolling with patience for loading
        print("Scrolling to load more images...")
        previous_count = 0
        no_new_images_count = 0
        max_scrolls = 50  # Increased from 10
        
        for scroll in range(max_scrolls):
            # Scroll down
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)  # Wait for images to load
            
            # Check if we're getting new images
            current_images = page.query_selector_all("img")
            current_count = len(current_images)
            
            if current_count > previous_count:
                print(f"  Scroll {scroll+1}: Found {current_count} images")
                no_new_images_count = 0
                previous_count = current_count
            else:
                no_new_images_count += 1
                # If no new images for 3 consecutive scrolls, we might be done
                if no_new_images_count >= 3:
                    print(f"  No new images found after {no_new_images_count} scrolls, stopping")
                    break
        
        # Final wait and get all images
        time.sleep(2)
        image_elements = page.query_selector_all("img")
        print(f"Found {len(image_elements)} total images")
        
        # Filter for Pinterest pin images (avoid UI elements, avatars, etc.)
        valid_images = []
        for el in image_elements:
            url = el.get_attribute("src")
            if url and any(size in url for size in ["236x", "474x", "564x", "736x"]):
                # Pinterest pin images typically have these size indicators
                valid_images.append(el)
        
        print(f"Found {len(valid_images)} valid pin images")

        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)

        new_count = 0
        for el in tqdm(valid_images, desc="Processing images"):
            url = el.get_attribute("src")
            alt = el.get_attribute("alt")

            if url and url not in metadata:
                filename = generate_filename(url)
                success = download_image(url, filename)
                if success:
                    metadata[url] = {
                        "filename": filename,
                        "alt": alt,
                        "timestamp": time.time(),
                    }
                    new_count += 1

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        browser.close()
        print(f"Downloaded {new_count} new images.")
        print(f"Total images in metadata: {len(metadata)}")

if __name__ == "__main__":
    ensure_dir()
    
    # Use default board or ask for input
    board_url = input(f"Enter Pinterest board URL (press Enter for default: {DEFAULT_BOARD_URL}): ").strip()
    
    if not board_url:
        board_url = DEFAULT_BOARD_URL
        print(f"Using default board: {board_url}")
    
    scrape_images(board_url)
