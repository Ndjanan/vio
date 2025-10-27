
Scholar-Spot Adaptation README
Generated: 2025-09-26T18:45:22.197185 UTC

Added files (backend/integration):
- supabaseClient.js : Node.js wrapper to connect to Supabase.
- recommendation_service.py : Python scaffold for hybrid recommendations (ML + rules).
- supabase_schema.sql : SQL to create the recommended tables in Supabase.

What I changed/added:
- Kept your existing frontend (client/) as-is.
- Added backend integration scaffolding to connect your model with Supabase.
- Added a detailed PDF 'DOC_Scholar-Spot_Adaptation.pdf' with architecture, Supabase table definitions, deployment steps, and how to train/deploy the hybrid model.

Next steps for you:
1) Set SUPABASE_URL and SUPABASE_KEY in your backend environment.
2) Run the SQL in supabase_schema.sql in your Supabase SQL editor.
3) Adapt recommendation_service.py feature engineering to match your model inputs.
4) Connect frontend to backend endpoints to fetch recommendations from 'recommendations' table.

This is a scaffold and should be tested locally before production deployment.

--- Updated instructions: Added FastAPI service and Node.js endpoint integration ---

1) FastAPI service (Python)
- Location: backend/api/fastapi_app.py
- Requirements: backend/api/requirements.txt
- Run locally: install requirements (pip install -r requirements.txt), then:
  uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
- Ensure SUPABASE_URL & SUPABASE_KEY are set in env before running (or the service will fallback to local CSV for scholarships).

2) Node.js route
- Location: backend/routes/recommendations.js
- This route calls the FastAPI /predict endpoint and returns recommendations to the frontend.
- Ensure FASTAPI_URL env var points to your FastAPI service (e.g., http://localhost:8000)

3) End-to-end flow
- Start Supabase (set env vars), start FastAPI, then start Node.js backend.
- Frontend should call /api/recommendations with {user_id: '<user-uuid>'} to get recommendations.
