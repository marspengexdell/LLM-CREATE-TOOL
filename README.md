<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This repository now contains a minimal FastAPI backend with a React/Vite front-end for orchestrating multimodal workflows. The backend provides dataset and workflow management APIs as well as a stubbed execution engine that produces placeholder outputs server-side so the UI can visualise runs without external dependencies.

## Prerequisites

- Node.js 18+
- Python 3.10+
- (Optional) Docker & Docker Compose

## Environment configuration

- Generate a new **GEMINI_API_KEY** in your Google Cloud project and revoke any previously exposed keys.
- Provide the key to the application via environment variables instead of committing it to the repository.
  - For local development you can create an `.env` file (ignored by Git) containing `GEMINI_API_KEY=<your-new-key>`.
  - Alternatively, export the variable in your shell session before starting the services: `export GEMINI_API_KEY=<your-new-key>`.
- In automated environments (Docker, CI/CD, hosting providers), inject the variable using the platform's secret manager or runtime configuration features.

## Backend setup

1. Provide a `GEMINI_API_KEY` environment variable (e.g. via your shell or Docker Compose overrides). The key is **not** stored in the repository—create a local `.env` file if desired, but keep it out of source control.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server:

2. Run the API server:
 main
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


Uploaded datasets are written to per-dataset folders inside `storage/datasets/` along with a JSON metadata file. Workflow definitions are persisted to `storage/workflows/`, and execution logs are written to `storage/logs/backend.log`.

### Dataset uploads

`POST /api/v1/datasets/upload` accepts a multipart form field named `file`. Uploads are restricted to the following extensions: `.bmp`, `.csv`, `.gif`, `.jpeg`, `.jpg`, `.json`, `.md`, `.png`, `.txt`, and `.webp`. The API responds with the stored dataset metadata, including the generated dataset `id` and a storage `path` (relative to the backend `storage/` directory) alongside the original filename, size, MIME type, and preview data when available.

Uploaded datasets and workflow definitions are stored under `storage/datasets/` and `storage/workflows/` respectively. Execution logs are written to `storage/logs/backend.log`.


### Error responses

All API errors use a consistent envelope so the frontend can reliably surface issues to end users. Error responses have the shape:

```json
{
  "error": {
    "code": 400,
    "message": "Uploaded file must include a filename.",
    "details": {
      "errorCode": "DATASET_MISSING_FILENAME"
    }
  }
}
```

- `code` is the HTTP status code of the response.
- `message` is a human readable description.
- `details` (optional) includes a stable `errorCode` and any additional structured context to help the UI decide how to react.

Unhandled exceptions are logged server-side and returned as HTTP 500 with an `INTERNAL_SERVER_ERROR` code inside `details`.

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

### Development

To iterate locally with hot-reload enabled, start the stack with the default Compose file:

```bash
docker compose up --build
```

- Backend: exposed at `http://localhost:8000`
- Frontend dev server: `http://localhost:3000`

Both services mount the repository directory (`.:/app` in `docker-compose.yml`) so code changes are reflected without rebuilding. Place your development secrets—at minimum `GEMINI_API_KEY=<value>`—in an `.env` file at the project root so the backend container picks them up via the `env_file` directive.

### Production

For production deployments you typically want to run the built images without the development bind mounts. Create a production override (for example, `docker-compose.prod.yml`) that removes the `.:/app` volumes and sets any production-only configuration, or add a `production` profile to your Compose file. Then launch the stack in detached mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

Ensure your production environment provides the same required variables—especially `GEMINI_API_KEY`—either through an override `env_file` (such as `.env.production`) or your orchestration platform's secret manager before starting the services.

## Deployment notes

- Use the FastAPI app (`main:app`) behind a production-ready ASGI server such as Uvicorn or Gunicorn.
- Frontend production builds can be created with `npm run build` and served via a static host or reverse proxy (e.g., Nginx) that forwards `/api` traffic to the backend service.
- Persist the `storage/` directory in production to retain datasets, workflows, and logs.

## Smoke tests

After completing the backend setup, ensure the API is running at `http://localhost:8000`. The commands below assume `curl` (and optionally [`jq`](https://stedolan.github.io/jq/)) is available on your PATH.

### Prepare sample assets

```bash
mkdir -p smoke-tests
cat <<'EOF' > smoke-tests/dataset.csv
id,value
1,example
EOF

cat <<'EOF' > smoke-tests/workflow.json
{
  "name": "Smoke test flow",
  "description": "Minimal workflow used in README smoke tests.",
  "nodes": [
    {
      "id": "text-input-1",
      "type": "text_input",
      "position": {"x": 0, "y": 0},
      "data": {"text": "Hello from smoke test"}
    },
    {
      "id": "decision-1",
      "type": "decision",
      "position": {"x": 240, "y": 0},
      "data": {"condition": "input != ''"}
    },
    {
      "id": "output-1",
      "type": "text_output",
      "position": {"x": 480, "y": 0},
      "data": {}
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "fromNode": "text-input-1",
      "fromPort": "out",
      "toNode": "decision-1",
      "toPort": "input"
    },
    {
      "id": "edge-2",
      "fromNode": "decision-1",
      "fromPort": "true",
      "toNode": "output-1",
      "toPort": "in"
    }
  ]
}
EOF
```

### API checks

```bash
# GET /api/v1/models
curl -sS http://localhost:8000/api/v1/models | jq
# → [
#     {
#       "id": "gemini-1.5-flash",
#       "name": "Gemini 1.5 Flash",
#       "description": "Fast multimodal model for interactive workflows."
#     },
#     {
#       "id": "gemini-1.5-pro",
#       "name": "Gemini 1.5 Pro",
#       "description": "Higher quality Gemini model suited for complex reasoning tasks."
#     }
#   ]

# GET /api/v1/datasets (before upload)
curl -sS http://localhost:8000/api/v1/datasets | jq
# → []  # if no datasets have been uploaded yet

# POST /api/v1/datasets/upload
curl -sS -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@smoke-tests/dataset.csv" | jq
# → {
#     "id": "2f0e8f5c-...",
#     "name": "dataset.csv",
#     "filename": "2f0e8f5c-....csv",
#     "path": "datasets/2f0e8f5c-....csv",
#     "size": 19,
#     "type": "text",
#     "mimeType": "text/csv",
#     "preview": "id,value\\n1,example\\n",
#     "uploadedAt": "2024-06-01T12:34:56Z"
#   }

# GET /api/v1/datasets (after upload)
curl -sS http://localhost:8000/api/v1/datasets | jq
# → [
#     {
#       "id": "2f0e8f5c-...",
#       "datasetId": "2f0e8f5c-...",
#       "name": "dataset.csv",
#       "storedFilename": "dataset.csv",
#       "size": 19,
#       "type": "text",
#       "mimeType": "text/csv",
#       "preview": "id,value\\n1,example\\n",
#       "uploadedAt": "2024-06-01T12:34:56Z",
#       "storagePath": "storage/datasets/2f0e8f5c-.../dataset.csv"
#     }
#   ]

# GET /api/v1/workflows
curl -sS http://localhost:8000/api/v1/workflows | jq
# → []  # if no workflows have been saved yet

# POST /api/v1/workflows/save
curl -sS -X POST http://localhost:8000/api/v1/workflows/save \
  -H "Content-Type: application/json" \
  -d @smoke-tests/workflow.json | jq
# → {
#     "id": "8b3f6c2d-..."
#   }

# GET /api/v1/workflows (after save)
curl -sS http://localhost:8000/api/v1/workflows | jq
# → [
#     {
#       "id": "8b3f6c2d-...",
#       "name": "Smoke test flow",
#       "description": "Minimal workflow used in README smoke tests.",
#       "updatedAt": "2024-06-01T12:35:10Z"
#     }
#   ]

# POST /api/v1/workflow/run
curl -sS -X POST http://localhost:8000/api/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d @smoke-tests/workflow.json | jq
# → {
#     "run_id": "c7191a80-...",
#     "nodes": [
#       {
#         "id": "text-input-1",
#         "type": "text_input",
#         "x": 0.0,
#         "y": 0.0,
#         "data": {"text": "Hello from smoke test"},
#         "status": "done"
#       },
#       {
#         "id": "decision-1",
#         "type": "decision",
#         "x": 240.0,
#         "y": 0.0,
#         "data": {"condition": "input != ''", "decision": true},
#         "status": "done"
#       },
#       {
#         "id": "output-1",
#         "type": "text_output",
#         "x": 480.0,
#         "y": 0.0,
#         "data": {},
#         "status": "done"
#       }
#     ]
#   }
```

When you are finished, you can remove the temporary files with `rm -r smoke-tests`.

## Testing

The frontend includes Vitest suites that cover critical workflow interactions such as uploading datasets, saving/loading workflows, and executing a run. To install dependencies and run the tests locally:

```bash
npm install
npm run test
```

Vitest will use a mocked backend (via MSW) so no API services need to be running for these checks.
