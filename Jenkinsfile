pipeline {
  agent any

  environment {
    DOCKER_CREDS = credentials('docker-registry-creds')
    REGISTRY_URL = "docker.io"
    ORCH_IMAGE = "orchestrator-agent"
    RAG_IMAGE  = "rag-agent"
    IMAGE_TAG  = "ci-${BUILD_NUMBER}"
    TAG_FROM_GIT = "${env.TAG_NAME ?: env.GIT_BRANCH}"
  }

  stages {
    stage('Checkout') {
      agent any
      steps {
        checkout scm
        sh 'git fetch --tags --force'
        script {
          env.ORCH_IMAGE = "${DOCKER_CREDS_USR}/orchestrator-agent"
          env.RAG_IMAGE  = "${DOCKER_CREDS_USR}/rag-agent"
        }
      }
    }

    stage('Test orchestrator-agent') {
      steps {
        sh '''
          cd src/agents/orchestrator-agent
          docker build -t ${ORCH_IMAGE}:${IMAGE_TAG} -f Dockerfile .
        '''
      }
    }

    stage('Test rag-agent') {
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
          echo ${DOCKER_CREDS_PSW} | docker login ${REGISTRY_URL} \
            -u ${DOCKER_CREDS_USR} --password-stdin
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
  }

  post {
    always {
      cleanWs()
    }
  }
}
