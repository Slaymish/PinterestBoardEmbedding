# Cluster Categories Configuration

This project uses CSV files to configure the categories used for dynamic cluster labeling. This makes it easy to add new categories without modifying the code.

## Configuration Files

### `cluster_categories.csv`
Contains the broad categories that are tested first during cluster discovery.

**Columns:**
- `category_type`: A human-readable grouping (e.g., "Home & Decor", "Fashion & Beauty")  
- `category_name`: The actual category text used for CLIP similarity matching

**Example:**
```csv
category_type,category_name
Home & Decor,furniture and home decor
Home & Decor,cozy hygge style
Fashion & Beauty,hairstyles
```

### `cluster_refinements.csv`
Maps broad categories to more specific subcategories for refinement.

**Columns:**
- `broad_category`: The broad category name (must match a `category_name` from `cluster_categories.csv`)
- `refined_category`: More specific category to test if the broad category matches well

**Example:**
```csv
broad_category,refined_category
hairstyles,wolf haircuts
hairstyles,pixie cuts
hairstyles,bob haircuts
```

## Adding New Categories

### To add a new broad category:
1. Open `cluster_categories.csv`
2. Add a new row with an appropriate `category_type` and unique `category_name`
3. Save the file

### To add refinements for a category:
1. Open `cluster_refinements.csv` 
2. Add rows mapping your `broad_category` to specific `refined_category` values
3. Save the file

### Examples of categories you might add:
- `gay cute twinks` (under Lifestyle & People)
- `wolf haircut` (under Fashion & Beauty, refined from hairstyles)
- `hammock indoors` (under Unique & Niche Concepts)
- `dark academia aesthetic` (under Aesthetics & Trends)

## How It Works

1. **Broad Discovery**: The system tests all categories from `cluster_categories.csv` against sample images using CLIP similarity
2. **Refinement**: If refinements exist in `cluster_refinements.csv` for the best broad category, those are tested
3. **Variations**: Finally, stylistic variations (modern, vintage, cute, etc.) are automatically generated and tested
4. **Selection**: The category with the highest similarity score becomes the cluster label

The system is fully dynamic - no code changes needed to add new categories!
