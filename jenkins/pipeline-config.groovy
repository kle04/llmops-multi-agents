/**
 * Jenkins Pipeline Configuration for LLMOps Multi-Agent System
 * This file contains reusable pipeline configurations and utilities
 */

// Pipeline configuration constants
def getPipelineConfig() {
    return [
        // Timeout settings
        PIPELINE_TIMEOUT: 60,
        STAGE_TIMEOUT: 30,
        
        // Cache settings  
        MAX_CACHE_FILES: 5,
        CACHE_RETENTION_DAYS: 7,
        
        // Docker settings
        DOCKER_TIMEOUT: 300,
        QDRANT_HEALTH_CHECK_RETRIES: 30,
        QDRANT_HEALTH_CHECK_INTERVAL: 5,
        
        // Model settings
        DEFAULT_EMBEDDING_MODEL: "AITeamVN/Vietnamese_Embedding_v2",
        DEFAULT_BATCH_SIZE: 32,
        DEFAULT_COLLECTION: "mental_health_vi",
        
        // Directory structure
        DIRECTORIES: [
            RAW_DATA: "data/raw",
            PROCESSED_TEXT: "data/processed/text", 
            PROCESSED_CHUNKS: "data/processed/chunks",
            EMBEDDINGS: "data/embeddings",
            CACHE: "cache"
        ]
    ]
}

// Utility functions for pipeline
def calculateFileHash(String path, String pattern = "*") {
    """
    Calculate MD5 hash of files in a directory
    """
    return sh(
        script: "find ${path} -name '${pattern}' -exec md5sum {} \\; | sort | md5sum | cut -d' ' -f1",
        returnStdout: true
    ).trim()
}

def waitForQdrant(String url, int maxRetries = 30, int interval = 5) {
    """
    Wait for Qdrant service to be ready
    """
    sh """
        echo "⏳ Waiting for Qdrant at ${url}..."
        for i in {1..${maxRetries}}; do
            if curl -s ${url}/health >/dev/null 2>&1; then
                echo "✅ Qdrant is ready"
                exit 0
            fi
            echo "⏳ Waiting... (\$i/${maxRetries})"
            sleep ${interval}
        done
        echo "❌ Qdrant failed to start within timeout"
        exit 1
    """
}

def archiveWithRetention(String artifacts, int maxFiles = 5) {
    """
    Archive artifacts with automatic cleanup of old files
    """
    archiveArtifacts artifacts: artifacts, allowEmptyArchive: true
    
    // Clean up old artifacts (implementation depends on Jenkins setup)
    sh """
        # This would typically be handled by Jenkins built-in retention policies
        echo "📦 Archived: ${artifacts}"
    """
}

def sendNotification(String status, Map details = [:]) {
    """
    Send pipeline status notification
    """
    def color = status == 'SUCCESS' ? 'good' : 'danger'
    def emoji = status == 'SUCCESS' ? '🎉' : '❌'
    
    def message = """
${emoji} LLMOps Pipeline ${status}
📊 Build: ${env.BUILD_NUMBER}
⏱️ Duration: ${details.duration ?: 'N/A'}
🌟 Branch: ${env.BRANCH_NAME ?: 'main'}
"""
    
    if (details.cacheStats) {
        message += """
📦 Cache Stats:
  - Preprocessing: ${details.cacheStats.preprocessing}
  - Embedding: ${details.cacheStats.embedding}
"""
    }
    
    if (status == 'FAILURE') {
        message += """
🔍 Logs: ${env.BUILD_URL}console
"""
    }
    
    echo message
    
    // Uncomment and configure for actual notifications
    // slackSend(color: color, message: message, channel: '#llmops')
    // emailext(subject: "LLMOps Pipeline ${status}", body: message, to: 'team@company.com')
}

def validateEnvironment() {
    """
    Validate required environment and dependencies
    """
    sh """
        echo "🔍 Validating environment..."
        
        # Check Python version
        python --version
        
        # Check Docker
        docker --version
        docker-compose --version
        
        # Check required directories
        mkdir -p data/raw data/processed/text data/processed/chunks data/embeddings cache
        
        # Check disk space (minimum 10GB free)
        AVAILABLE=\$(df ${WORKSPACE} | awk 'NR==2 {print \$4}')
        if [ \$AVAILABLE -lt 10485760 ]; then
            echo "❌ Insufficient disk space: \${AVAILABLE}KB available, 10GB required"
            exit 1
        fi
        
        echo "✅ Environment validation passed"
    """
}

// Export functions for use in Jenkinsfile
return this
