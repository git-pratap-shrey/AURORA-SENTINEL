cd frontend
npm run build
cd ..

.\venv\Scripts\python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

cloudflared tunnel --url http://localhost:8000
