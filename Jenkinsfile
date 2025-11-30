pipeline {
  agent any

  environment {
    DOCKER_CREDS     = credentials('docker-registry-creds')
    GITHUB_TOKEN     = credentials('github-pat')
    TELEGRAM_TOKEN   = credentials('telegram-bot-token')
    TELEGRAM_CHAT_ID = "1006808421"

    REGISTRY_URL = "docker.io"
    ORCH_IMAGE = "orchestrator-agent"
    RAG_IMAGE  = "rag-agent"

    IMAGE_TAG  = "ci-${BUILD_NUMBER}"
    TAG_FROM_GIT = "${env.TAG_NAME ?: env.GIT_BRANCH}"

    MANIFEST_REPO = "https://github.com/KhanhLe04/manifest-llmops-multi-agents.git"
    BRANCH_NAME   = "bot/ci-${TAG_FROM_GIT}"
  }

  stages {

    stage('Checkout') {
      steps {
        checkout scm
        sh 'git fetch --tags --force'
        script {
          env.ORCH_IMAGE = "${DOCKER_CREDS_USR}/orchestrator-agent"
          env.RAG_IMAGE  = "${DOCKER_CREDS_USR}/rag-agent"
        }
      }
    }

    stage('Lint test') {
      steps {
        sh '''
          echo "=========== Lint test for Orchestrator Agent ==========="
          flake8 src/agents/orchestrator-agent
          echo "=========== Lint test for RAG Agent ==========="
          flake8 src/agents/rag-agent
        '''
      }
    }

    stage('Build orchestrator-agent') {
      steps {
        sh '''
          cd src/agents/orchestrator-agent
          docker build -t ${ORCH_IMAGE}:${IMAGE_TAG} -f Dockerfile .
        '''
      }
    }

    stage('Build rag-agent') {
      steps {
        sh '''
          cd src/agents/rag-agent
          docker build -t ${RAG_IMAGE}:${IMAGE_TAG} -f Dockerfile .
        '''
      }
    }

    stage('Tag & Push images') {
      when {
        buildingTag()
      }
      steps {
        sh """
          echo ${DOCKER_CREDS_PSW} | docker login ${REGISTRY_URL} -u ${DOCKER_CREDS_USR} --password-stdin

          docker tag ${ORCH_IMAGE}:${IMAGE_TAG} ${DOCKER_CREDS_USR}/${ORCH_IMAGE}:${TAG_FROM_GIT}
          docker tag ${RAG_IMAGE}:${IMAGE_TAG} ${DOCKER_CREDS_USR}/${RAG_IMAGE}:${TAG_FROM_GIT}

          docker push ${DOCKER_CREDS_USR}/${ORCH_IMAGE}:${TAG_FROM_GIT}
          docker push ${DOCKER_CREDS_USR}/${RAG_IMAGE}:${TAG_FROM_GIT}

          docker rmi ${ORCH_IMAGE}:${IMAGE_TAG}
          docker rmi ${RAG_IMAGE}:${IMAGE_TAG}
          docker rmi ${DOCKER_CREDS_USR}/${ORCH_IMAGE}:${TAG_FROM_GIT}
          docker rmi ${DOCKER_CREDS_USR}/${RAG_IMAGE}:${TAG_FROM_GIT}
        """
      }
    }

    stage('Create PR for Manifest Update') {
      when {
        buildingTag()
      }
      steps {
        sh """
          rm -rf manifest-repo
          git clone ${MANIFEST_REPO} manifest-repo
          cd manifest-repo

          git checkout -b ${BRANCH_NAME}

          sed -i 's|  tag:.*|  tag: "'${TAG_FROM_GIT}'"|' charts/orchestrator-agent/values.yaml
          sed -i 's|  tag:.*|  tag: "'${TAG_FROM_GIT}'"|' charts/rag-agent/values.yaml

          git config user.name "jenkins-bot"
          git config user.email "ci-bot@example.com"

          git add .
          git commit -m "Update images to tag ${TAG_FROM_GIT}"
          git push https://${GITHUB_TOKEN}@github.com/KhanhLe04/manifest-llmops-multi-agents.git ${BRANCH_NAME}

          gh pr create \
            --repo KhanhLe04/manifest-llmops-multi-agents \
            --title "Update images to ${TAG_FROM_GIT}" \
            --body "CI/CD: Update orchestrator-agent và rag-agent lên tag **${TAG_FROM_GIT}**" \
            --base main \
            --head ${BRANCH_NAME}
        """
      }
    }

    stage('Send Telegram Notification') {
      steps {
        script {
          def msg = """
          CI/CD Pipeline Completed

          Tag: ${TAG_FROM_GIT}
          Images:
          - orchestrator-agent:${TAG_FROM_GIT}
          - rag-agent:${TAG_FROM_GIT}

          Pull Requests:
          https://github.com/KhanhLe04/manifest-llmops-multi-agents/pulls

          Pipeline:
          ${env.BUILD_URL}
          """

          sh """
            curl -X POST \
            -H 'Content-Type: application/json' \
            -d '{"chat_id":"${TELEGRAM_CHAT_ID}","text":"${msg}"}' \
            https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage
          """
        }
      }
    }
  }

  post {
    always {
      cleanWs()
    }
  }
}