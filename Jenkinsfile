pipeline {
  agent none

  environment {
    DOCKER_CREDS = credentials('docker-registry-creds')
    REGISTRY_URL = "docker.io"
    ORCH_IMAGE = "orchestrator-agent"
    RAG_IMAGE  = "rag-agent"
    IMAGE_TAG  = ""
    TAG_FROM_GIT = ""
  }

  stages {
    stage('Checkout') {
      agent any
      steps {
        checkout scm
        sh 'git fetch --tags --force'
        script {
          env.IMAGE_TAG = sh(returnStdout: true,
              script: "git rev-parse --short HEAD").trim()
          env.TAG_FROM_GIT = sh(returnStdout: true,
              script: 'git describe --tags --exact-match HEAD 2>/dev/null || true').trim()
          env.ORCH_IMAGE = "${DOCKER_CREDS_USR}/orchestrator-agent"
          env.RAG_IMAGE  = "${DOCKER_CREDS_USR}/rag-agent"
        }
      }
    }

    stage('Test orchestrator-agent') {
      agent {
        kubernetes {
          defaultContainer 'docker'
          yaml """
          apiVersion: v1
          kind: Pod
          spec:
            restartPolicy: Never
            containers:
              - name: docker
                image: docker:29.0
                command:
                  - cat
                tty: true
                volumeMounts:
                  - name: docker-sock
                    mountPath: /var/run/docker.sock
            volumes:
              - name: docker-sock
                hostPath:
                  path: /var/run/docker.sock
          """
        }
      }
      steps {
        container('docker') {
          sh '''
            cd src/agents/orchestrator-agent
            docker build -t ${ORCH_IMAGE}:${IMAGE_TAG} -f Dockerfile .
          '''
        }
      }
    }

    stage('Test rag-agent') {
      agent {
        kubernetes {
          defaultContainer 'docker'
          yaml """
          apiVersion: v1
          kind: Pod
          spec:
            restartPolicy: Never
            containers:
              - name: docker
                image: docker:29.0
                command:
                  - cat
                tty: true
                volumeMounts:
                  - name: docker-sock
                    mountPath: /var/run/docker.sock
            volumes:
              - name: docker-sock
                hostPath:
                  path: /var/run/docker.sock
          """
        }
      }
      steps {
        container('docker') {
          sh '''
            cd src/agents/rag-agent
            docker build -t ${RAG_IMAGE}:${IMAGE_TAG} -f Dockerfile .
          '''
        }
      }
    }

    stage('Tag & Push images (on git tag)') {
      when {
        expression { env.TAG_FROM_GIT?.trim() }
      }
      agent {
        kubernetes {
          defaultContainer 'docker'
          yaml """
          apiVersion: v1
          kind: Pod
          spec:
            restartPolicy: Never
            containers:
              - name: docker
                image: docker:29.0
                command:
                  - cat
                tty: true
                volumeMounts:
                  - name: docker-sock
                    mountPath: /var/run/docker.sock
            volumes:
              - name: docker-sock
                hostPath:
                  path: /var/run/docker.sock
          """
        }
      }
      steps {
        container('docker') {
          sh """
            echo ${DOCKER_CREDS_PSW} | docker login ${REGISTRY_URL} \
              -u ${DOCKER_CREDS_USR} --password-stdin
            docker tag ${ORCH_IMAGE}:${IMAGE_TAG} ${DOCKER_CREDS_USR}/${ORCH_IMAGE}:${TAG_FROM_GIT}
            docker tag ${RAG_IMAGE}:${IMAGE_TAG} ${DOCKER_CREDS_USR}/${RAG_IMAGE}:${TAG_FROM_GIT}
            docker push ${DOCKER_CREDS_USR}/${ORCH_IMAGE}:${TAG_FROM_GIT}
            docker push ${DOCKER_CREDS_USR}/${RAG_IMAGE}:${TAG_FROM_GIT}
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
