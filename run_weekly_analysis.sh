#!/bin/bash

# Weekly Pinterest Aesthetic Analysis Script
# This script activates the virtual environment and runs the complete pipeline
# to track your aesthetic evolution over time.

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/home/hamish/Documents/Projects/PinterestBoardEmbedding"
VENV_PATH="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/weekly_analysis.log"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log "========================================="
log "Starting weekly Pinterest analysis..."
log "========================================="

# Change to project directory
cd "$PROJECT_DIR" || {
    log "ERROR: Could not change to project directory: $PROJECT_DIR"
    exit 1
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log "ERROR: Virtual environment not found at: $VENV_PATH"
    log "Please create it with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_PATH/bin/activate" || {
    log "ERROR: Could not activate virtual environment"
    exit 1
}

# Check if main.py exists
if [ ! -f "main.py" ]; then
    log "ERROR: main.py not found in $PROJECT_DIR"
    exit 1
fi

# Run the complete pipeline
log "Running complete Pinterest analysis pipeline..."
log "This will: scrape → cluster → analyze aesthetic evolution"

# Capture output and errors
if python main.py --full >> "$LOG_FILE" 2>&1; then
    log "✅ Weekly analysis completed successfully!"
    
    # Show current status
    log "Current collection status:"
    python main.py --status >> "$LOG_FILE" 2>&1
    
    # Check if summary was generated
    if [ -f "pinterest_images/summary.md" ]; then
        log "📊 Evolution report generated: pinterest_images/summary.md"
        log "--- Summary Preview ---"
        head -20 "pinterest_images/summary.md" >> "$LOG_FILE" 2>&1
        log "--- End Preview ---"
    fi
    
else
    log "❌ Weekly analysis failed! Check the log above for details."
    exit 1
fi

# Deactivate virtual environment
deactivate

log "Weekly Pinterest analysis completed at $(date)"
log "========================================="

# Optional: Clean up old log entries (keep last 30 days)
if command -v tail >/dev/null 2>&1; then
    # Keep only last 1000 lines of log file
    tail -1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi

echo "✅ Weekly analysis complete! Check $LOG_FILE for details."
