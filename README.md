<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This repository now contains a minimal FastAPI backend with a React/Vite front-end for orchestrating multimodal workflows. The backend provides dataset and workflow management APIs as well as a stubbed execution engine that produces placeholder outputs server-side so the UI can visualise runs without external dependencies.

## Prerequisites

- Node.js 18+
- Python 3.10+
- (Optional) Docker & Docker Compose

## Backend setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000` with the following routes:

- `GET /api/v1/models`
- `GET /api/v1/datasets`
- `POST /api/v1/datasets/upload`
- `GET /api/v1/workflows`
- `GET /api/v1/workflows/{id}`
- `POST /api/v1/workflows/save`
- `POST /api/v1/workflow/run`

Uploaded datasets and workflow definitions are stored under `storage/datasets/` and `storage/workflows/` respectively. Execution logs are written to `storage/logs/backend.log`.

## Frontend setup

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the Vite dev server:
   ```bash
   npm run dev
   ```

The app runs on `http://localhost:3000` and proxies `/api` requests to the backend (`http://localhost:8000` by default).

## Run with Docker Compose

To start both services with Docker Compose:

```bash
docker compose up --build
```

- Backend: exposed at `http://localhost:8000`
- Frontend dev server: `http://localhost:3000`

Volumes are mounted for live reloading during development. Provide environment variables through an `.env` file in the project root before running the stack.

## Deployment notes

- Use the FastAPI app (`main:app`) behind a production-ready ASGI server such as Uvicorn or Gunicorn.
- Frontend production builds can be created with `npm run build` and served via a static host or reverse proxy (e.g., Nginx) that forwards `/api` traffic to the backend service.
- Persist the `storage/` directory in production to retain datasets, workflows, and logs.

## Testing (future work)

Unit tests and CI pipelines are not yet configured. Recommended next steps include adding pytest coverage for the execution engine and Vitest/RTL coverage for critical frontend interactions.
