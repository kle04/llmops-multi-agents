# Orchestrator Agent REST API Documentation

## Overview
Base URL: `http://<host>:<port>` (Default: `http://localhost:7010`)

This API provides an interface for the Orchestrator Agent, handling user authentication, chat interactions, session management, and document ingestion.

## Authentication
The API uses **OAuth2 with Password Flow** and **JWT (JSON Web Tokens)**.
All secured endpoints require the `Authorization` header.

**Header Format:**
```http
Authorization: Bearer <your_access_token>
```

---

## 1. Authentication & Users

### Register New User
Create a new account.

*   **Endpoint**: `/register`
*   **Method**: `POST`
*   **Auth Required**: No

**Request Body:**
```json
{
  "username": "jdoe",
  "email": "jdoe@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
true
```

### Login (Get Token)
Authenticate and retrieve an access token.

*   **Endpoint**: `/token`
*   **Method**: `POST`
*   **Content-Type**: `application/x-www-form-urlencoded`
*   **Auth Required**: No

**Request Body (Form Data):**
*   `username`: Your username
*   `password`: Your password

**Response (200 OK):**
```json
{
  "access_token": "ey...<jwt_token>...",
  "token_type": "bearer"
}
```

### Get Current User
Retrieve profile information for the authenticated user.

*   **Endpoint**: `/users/me`
*   **Method**: `GET`
*   **Auth Required**: Yes

**Response (200 OK):**
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "jdoe",
  "email": "jdoe@example.com",
  "created_at": "2023-10-27T10:00:00Z"
}
```

---

## 2. Chat Operations

### Send Message
Send a message to the Orchestrator Agent. The agent will route the request to the appropriate sub-agent or handle it directly.

*   **Endpoint**: `/chat`
*   **Method**: `POST`
*   **Auth Required**: Yes

**Request Body:**
```json
{
  "message": "How do I reset my password in Jenkins?",
  "session_id": "optional-uuid-string" 
}
```
*Note: `user_id` is automatically inferred from the token. providing `session_id` allows resuming a conversation. If omitted, a new session is created.*

**Response (200 OK):**
```json
{
  "selected_agent": "Orchestrator",
  "response": "To reset your Jenkins password, you can...",
  "sources": ["jenkins-guide.pdf"],
  "error": null
}
```

**Error (403 Forbidden):**
Returned if you attempt to use a `session_id` belonging to another user.

---

## 3. Session & History Management

### List User Sessions
Get a list of all chat sessions for the current user.

*   **Endpoint**: `/sessions/{user_id}`
*   **Method**: `GET`
*   **Auth Required**: Yes

**Response (200 OK):**
```json
{
  "sessions": [
    {
      "session_id": "550e8400-...",
      "last_updated": "2023-10-27T10:30:00Z"
    },
    ...
  ]
}
```

### Get Long-Term History
Retrieve the full chat history for a specific session from persistent storage (PostgreSQL).

*   **Endpoint**: `/history/{user_id}/{session_id}`
*   **Method**: `GET`
*   **Auth Required**: Yes

**Response (200 OK):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hi",
      "timestamp": "..."
    },
    {
      "role": "assistant",
      "content": "Hello!",
      "timestamp": "..."
    }
  ],
  "created_at": "...",
  "last_updated": "..."
}
```

### Get Short-Term History (Redis)
Retrieve recent context from fast memory.

*   **Endpoint**: `/history/redis/{user_id}/{session_id}`
*   **Method**: `GET`
*   **Query Params**: `limit` (int, optional)
*   **Auth Required**: Yes

### Delete Session
Permanently delete a chat session and its history.

*   **Endpoint**: `/history/{user_id}/{session_id}`
*   **Method**: `DELETE`
*   **Auth Required**: Yes

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Session ... deleted",
  "postgres_deleted": true
}
```

---

## 4. Knowledge Base (Ingestion)

### Ingest Document
Upload a document to the RAG knowledge base.

*   **Endpoint**: `/ingest`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`
*   **Auth Required**: Yes

**Request Body (Form Data):**
*   `file`: (File Object) Supported: `.pdf`, `.md`, `.txt`

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Successfully ingested document.pdf"
}
```

---

## 5. System Status

### Health Check
Check the status of the API and connected services (Redis, Postgres, Agents).

*   **Endpoint**: `/health`
*   **Method**: `GET`
*   **Auth Required**: No

**Response:**
```json
{
  "status": "healthy",
  "redis": { "connected": true },
  "postgres": { "connected": true },
  "agent": { ... },
  "ingestion": { ... }
}
```
