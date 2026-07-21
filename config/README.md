# Configuration

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned Python packages used by the project. |
| `Dockerfile` | CUDA 12.1 / PyTorch 2.4.1 image with system and Python dependencies. |
| `docker-compose.gpu.yml` | GPU-enabled service mounting the project at `/workspace`. |
| `run_gpu.sh` | Convenience wrapper that starts an interactive container. |

Run from the project root:

```bash
docker compose -f config/docker-compose.gpu.yml build
docker compose -f config/docker-compose.gpu.yml run --rm dual-translation
```
