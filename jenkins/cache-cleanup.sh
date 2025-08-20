#!/bin/bash
# Cache cleanup script for Jenkins pipeline
# Cleans up old cache files and maintains optimal storage usage

set -e

CACHE_DIR="${CACHE_DIR:-cache}"
MAX_CACHE_FILES="${MAX_CACHE_FILES:-5}"
MAX_AGE_DAYS="${MAX_AGE_DAYS:-7}"

echo "🧹 Starting cache cleanup..."
echo "📁 Cache directory: $CACHE_DIR"
echo "🔢 Max cache files: $MAX_CACHE_FILES"
echo "📅 Max age: $MAX_AGE_DAYS days"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Function to clean up by file count
cleanup_by_count() {
    local pattern="$1"
    local max_files="$2"
    
    echo "🗂️ Cleaning up $pattern files (keeping $max_files newest)..."
    
    # Get list of files sorted by modification time (newest first)
    local files=($(ls -t "$CACHE_DIR"/$pattern 2>/dev/null | tail -n +$((max_files + 1))))
    
    if [ ${#files[@]} -gt 0 ]; then
        echo "🗑️ Removing ${#files[@]} old cache files:"
        for file in "${files[@]}"; do
            echo "  - $file"
            rm -f "$CACHE_DIR/$file"
        done
    else
        echo "✅ No old cache files to remove"
    fi
}

# Function to clean up by age
cleanup_by_age() {
    local pattern="$1"
    local max_age="$2"
    
    echo "📅 Removing $pattern files older than $max_age days..."
    
    local count=$(find "$CACHE_DIR" -name "$pattern" -mtime +$max_age 2>/dev/null | wc -l)
    
    if [ $count -gt 0 ]; then
        echo "🗑️ Removing $count old cache files..."
        find "$CACHE_DIR" -name "$pattern" -mtime +$max_age -delete 2>/dev/null || true
    else
        echo "✅ No old cache files to remove by age"
    fi
}

# Function to get cache statistics
show_cache_stats() {
    echo "📊 Cache Statistics:"
    
    if [ -d "$CACHE_DIR" ]; then
        local total_files=$(find "$CACHE_DIR" -type f | wc -l)
        local total_size=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        local preprocessing_files=$(find "$CACHE_DIR" -name "preprocessing-*.tar.gz" | wc -l)
        local embedding_files=$(find "$CACHE_DIR" -name "embedding-*.tar.gz" | wc -l)
        
        echo "  📁 Total files: $total_files"
        echo "  💾 Total size: $total_size"
        echo "  🔄 Preprocessing caches: $preprocessing_files"
        echo "  🤖 Embedding caches: $embedding_files"
        
        # Show newest cache files
        echo "  📅 Recent cache files:"
        ls -lt "$CACHE_DIR"/*.tar.gz 2>/dev/null | head -5 | while read -r line; do
            echo "    $line"
        done
    else
        echo "  📁 Cache directory not found"
    fi
}

# Main cleanup process
main() {
    echo "🚀 Cache cleanup started at $(date)"
    
    # Show current state
    show_cache_stats
    
    # Clean up preprocessing caches
    cleanup_by_count "preprocessing-*.tar.gz" "$MAX_CACHE_FILES"
    cleanup_by_age "preprocessing-*.tar.gz" "$MAX_AGE_DAYS"
    
    # Clean up embedding caches  
    cleanup_by_count "embedding-*.tar.gz" "$MAX_CACHE_FILES"
    cleanup_by_age "embedding-*.tar.gz" "$MAX_AGE_DAYS"
    
    # Clean up any temporary files
    find "$CACHE_DIR" -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
    find "$CACHE_DIR" -name "*.lock" -mtime +1 -delete 2>/dev/null || true
    
    # Show final state
    echo ""
    echo "🎯 Final cache state:"
    show_cache_stats
    
    echo "✅ Cache cleanup completed at $(date)"
}

# Error handling
trap 'echo "❌ Cache cleanup failed at line $LINENO"' ERR

# Run main function
main "$@"
