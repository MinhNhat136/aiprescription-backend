# VoiceFillPrescription Backend

FastAPI backend for VoiceFillPrescription application with Whisper STT.

## Quick Start with Docker Compose

### Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.20+

### Setup

1. **Start all services (backend, frontend, database)**

   ```bash
   # From project root
   docker-compose up -d

   # View logs
   docker-compose logs -f
   ```

2. **Verify services are running**

   ```bash
   # Check service status
   docker-compose ps

   # Check backend health
   curl http://localhost:8000/health

   # Check frontend
   open http://localhost:5173
   ```

### Services

| Service | Port | Description |
|---------|------|-------------|
| backend | 8000 | FastAPI with Whisper STT |
| frontend | 5173 | Vite dev server with HMR |
| db | 5432 | PostgreSQL 17 |

### Environment Configuration

For Docker development, use the `.env.docker` template:

```bash
# Copy Docker environment file
cp .env.docker .env
```

**Key differences from local development:**
- `DEVICE=cpu` (no GPU in container)
- `DATABASE_URL` uses `db` hostname (Docker Compose service name)
- `HF_HOME=/models/huggingface` (persistent volume for Whisper models)

### Hot Reload

- **Backend**: `uvicorn --reload` is enabled. Code changes in `./backend/app` are reflected immediately.
- **Frontend**: Vite HMR is enabled. Code changes in `./frontend/src` are reflected immediately.

### Database

PostgreSQL data persists in a Docker volume (`postgres_data`). To reset:

```bash
# Stop services and remove volume
docker-compose down -v

# Restart (init.sql runs automatically)
docker-compose up -d
```

### Troubleshooting

**Backend won't start (Whisper model download)**
```bash
# First run downloads ~3GB model - may take 10+ minutes
docker-compose logs backend

# Increase timeout if needed in .env.docker
HF_HUB_DOWNLOAD_TIMEOUT=1200
```

**Database connection error**
```bash
# Wait for PostgreSQL to be ready
docker-compose ps db
docker-compose logs db
```

**Frontend can't connect to backend**
```bash
# Check backend is healthy
curl http://localhost:8000/health

# Check Vite proxy configuration
docker-compose exec frontend cat /app/vite.config.ts
```

### Development Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires PostgreSQL running)
uvicorn app.main:app --reload
```
