# Testing and Deployment Strategy

## 1. Introduction
This document outlines the strategy for ensuring the quality, reliability, and maintainability of the Visual Workflow Builder through a multi-layered testing approach and a containerized deployment plan.

---

## 2. Testing Strategy
Our testing strategy is divided into three layers: Unit Testing, Integration Testing, and End-to-End (E2E) Testing.

### 2.1. Unit Testing (Backend)
-   **Framework**: `pytest` for the Python/FastAPI backend.
-   **Objective**: To verify that individual components (e.g., the execution logic for a single node type) work correctly in isolation.
-   **Scope**:
    -   **Node Logic**: Each node's processing function will be tested with a variety of valid and invalid inputs. For example, testing the `decision_logic` node with conditions that evaluate to true, false, or raise errors.
    -   **API Endpoints**: Test each API endpoint's logic, such as request validation, data serialization, and error handling for `/api/v1/workflows/save`.
    -   **Utilities**: Any helper functions (data formatters, parsers) will have dedicated unit tests.
-   **Mocks**: External dependencies, especially the Google Gemini API, will be mocked to ensure tests are fast, deterministic, and do not incur costs.

### 2.2. Integration Testing (Backend)
-   **Framework**: `pytest`, using test fixtures to simulate workflow payloads.
-   **Objective**: To ensure that different components of the backend work together as expected, focusing on the `Execution Engine`.
-   **Scope**:
    -   **Topological Sort**: Test the execution order algorithm with various graph structures, including linear flows, parallel branches, and fan-in/fan-out patterns. Crucially, tests must verify that graphs with cycles are correctly identified and rejected.
    -   **Data Passing**: Simulate a full workflow run with mock data. Verify that the output from one node is correctly passed as input to the next node according to the defined edges.
    -   **Error Propagation**: Test the failure handling mechanism. When a node fails, verify that its status is set to `error` and all its downstream dependencies are correctly marked as `skipped`.

### 2.3. End-to-End (E2E) Testing
-   **Framework**: A browser automation framework like **Cypress** or **Playwright**.
-   **Objective**: To simulate real user scenarios from start to finish, ensuring the entire application stack (frontend, backend, APIs) works harmoniously.
-   **Key Scenarios**:
    1.  **Workflow Creation**: A test that drags multiple nodes from the toolbox, drops them on the canvas, and connects them in a specific order.
    2.  **Node Configuration**: A test that interacts with node bodies, such as uploading an image to an `image_input` node or selecting a model from the `model_hub` modal.
    3.  **Full Execution Run**: A test that builds a complete, valid workflow, clicks the "Run" button, and asserts that:
        -   The UI provides immediate "running" feedback.
        -   The final node statuses (`completed`, `error`) are correctly displayed.
        -   The output data (e.g., a report from the `generate_report` node) is visible in the UI.
    4.  **Workflow Management**: Tests for saving a workflow with a specific name, clearing the canvas, and then loading the same workflow back to verify its integrity.

---

## 3. Deployment Strategy
We will use **Docker** and **Docker Compose** to containerize the frontend and backend services, ensuring a consistent and reproducible deployment environment.

### 3.1. Overview
The application will be split into two main services:
-   `backend`: A container running the FastAPI application.
-   `frontend`: A lightweight Nginx container serving the static files from the built React application.

### 3.2. Backend Service (FastAPI)
The backend will be containerized using the following `Dockerfile`:

```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3. Frontend Service (React + Nginx)
We will use a multi-stage Docker build to create an optimized, lightweight image for the frontend.

```dockerfile
# Dockerfile.frontend

# --- Build Stage ---
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
# This assumes your build script is named "build"
RUN npm run build

# --- Serve Stage ---
FROM nginx:1.23-alpine
# Copy the built static files from the builder stage
COPY --from=builder /app/build /usr/share/nginx/html
# Copy a custom Nginx configuration to handle SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

A simple `nginx.conf` is needed to properly handle client-side routing:
```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;

    location / {
        root   /usr/share/nginx/html;
        index  index.html index.htm;
        # Redirect all non-file requests to index.html for the React router
        try_files $uri $uri/ /index.html;
    }

    # Reverse proxy for API calls
    location /api {
        proxy_pass http://backend:8000; # 'backend' is the service name in docker-compose
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 3.4. Orchestration with Docker Compose
A `docker-compose.yml` file will manage the services, networks, and environment variables.

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: workflow_backend
    env_file:
      - .env # Load environment variables from .env file
    ports:
      - "8000:8000" # Expose for direct access if needed, otherwise handled by frontend proxy
    networks:
      - app-network

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: workflow_frontend
    ports:
      - "8080:80" # Map host port 8080 to container port 80
    depends_on:
      - backend
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```
_Note: For production, the frontend would typically handle proxying `/api` requests to the backend container, so only the frontend port needs to be exposed to the public._

### 3.5. Environment Variable Management
-   Sensitive information, such as the `API_KEY` for the Gemini API, must not be hardcoded.
-   A `.env` file will be created in the project's root directory to store these variables.
-   **Example `.env` file:**
    ```
    # Environment variables for the backend service
    API_KEY="your-google-gemini-api-key-here"
    ```
-   This `.env` file will be loaded by the `backend` service in `docker-compose.yml`.
-   **Crucially, the `.env` file must be added to `.gitignore`** to prevent secrets from being committed to version control.
```
# .gitignore
.env
```
