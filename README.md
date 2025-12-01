# LLMOps - Multi-Agent Mental Health Support System

## Abstract

This project implements a scalable, microservice-based multi-agent system designed to provide automated, empathetic mental health support. By leveraging Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and a robust Orchestrator-Agent architecture, the system ensures that user queries are intelligently routed—either to a general conversationalist for emotional support or to a specialized knowledge agent for domain-specific advice grounded in verified psychological literature.

## 1. System Architecture

### 1.1. High-Level Overview

The system operates on a hub-and-spoke model where a central **Orchestrator Agent** manages user interactions and delegates tasks to specialized sub-agents. This architecture effectively solves the "agent sprawl" problem by providing a single, unified interface for the end-user while maintaining modularity in the backend.

![System Architecture](images/architecture_multi_agents.png)

### 1.2. Component Documentation

For a deep dive into the implementation details, algorithms, and workflows of each core component, please refer to their respective technical documentation:

*   **[Orchestrator Agent](./src/agents/orchestrator-agent/README.md)**  
    The central nervous system of the platform. It handles intent classification (Chitchat vs. Domain Inquiry), manages conversation state, and routes requests to downstream agents using the Agent-to-Agent (A2A) protocol.

*   **[Mental Health RAG Agent](./src/agents/rag-agent/README.md)**  
    A specialized worker agent built with **LangGraph**. It utilizes a "Retrieve-Grade-Generate" workflow to fetch context from the vector database, validate its relevance using LLM grading, and generate safe, empathetic responses.

*   **[Data Preparation Pipeline](./src/data-preparing/README.md)**  
    The automated ETL (Extract, Transform, Load) pipeline. It handles the ingestion of raw PDF documents, performs Vietnamese-specific text cleaning, chunks data for semantic context, and indexes vectors into **Qdrant**.

## 2. Technology Stack

The system is built on a modern, cloud-native stack designed for scalability and observability.

### 2.1. AI & Machine Learning
*   **Large Language Model**: Google Gemini API (Reasoning, Intent Classification, Generation)
*   **Embedding Model**: `intfloat/multilingual-e5-base` (Optimized for multilingual semantic search)
*   **Vector Database**: Qdrant (High-performance vector similarity search)
*   **Frameworks**: LangChain, LangGraph

### 2.2. Infrastructure & DevOps
| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Source Code** | GitHub | Version Control |
| **CI/CD** | Jenkins | Automated Testing & Deployment |
| **API Framework** | FastAPI | High-performance Async APIs |
| **Containerization** | Docker | Application Packaging |
| **Orchestration** | Kubernetes (K8s), Helm | Container Management & Scaling |
| **IaC** | Terraform | Infrastructure Provisioning (AWS/GCP) |
| **Monitoring** | Prometheus, Grafana, Loki | Metrics, Visualization, and Logging |
| **Ingress** | Nginx | Load Balancing & Traffic Routing |

## 3. Project Structure

```
llmops-multi-agents/
├── images/                  # Architecture diagrams and assets
├── src/
│   ├── agents/
│   │   ├── orchestrator-agent/  # Main entry point & router
│   │   └── rag-agent/           # Specialized knowledge worker
│   ├── data-preparing/          # ETL pipeline for knowledge base
│   └── evaluation               # Evaluation for Multi-agent system
└── README.md                # Project overview (this file)
```
