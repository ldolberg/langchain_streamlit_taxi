# Taxi SQL Chat — LangChain + Google Gemini + Postgres (Dockerized)

A dead-simple chat UI that turns natural language questions into PostgreSQL using Google Gemini via LangChain, runs the SQL against a Taxi dataset (NYC TLC), and visualizes the results. Everything is containerized with Docker Compose. Includes a lightweight ETL to download and load a real TLC month.

## Stack
- UI: Streamlit
- LLM & Orchestration: LangChain + `langchain-google-genai` (Gemini 1.5)
- Data: PostgreSQL (Docker) + NYC TLC Yellow Taxi data
- Data access: SQLAlchemy, `langchain_community.utilities.SQLDatabase`
- Charts: Altair
- Containerization: Docker + docker-compose
- CI/CD: GitHub Actions (Python sanity + Docker build)

## Why this stack?
- Rapid POC velocity: Streamlit keeps the UI minimal but powerful (charts, tabs, downloads).
- Good-enough NL→SQL: Gemini 1.5 Flash is fast and cost-effective; you can flip to Pro from the sidebar.
- Simple/portable data layer: Postgres with a tiny schema is easy to seed and reason about.
- LangChain primitives: `SQLDatabase` and a custom prompt keep the chain readable and debuggable.
- Docker-first: One command to bring up DB + app; ETL is a one-shot container or in-app button.

## Features
- Natural language to SQL generation against `yellow_trips`.
- Schema-aware prompting and display.
- Automatic SQL error recovery: on failure, the DB error is sent back to Gemini to fix and retry.
- Sidebar "Hint for the model" you can inject into prompts.
- Model and temperature controls (Gemini Flash/Pro).
- Metrics: rows/columns/elapsed, and optional EXPLAIN plan.
- Tabs: Data, Chart (auto line/bar), SQL, Summary (LLM), Download (CSV/Parquet).
- In-app ETL: download and load a month from NYC TLC.
- Quota handling: if Gemini rate/quota is exceeded, a clear error is shown and execution halts.

## Project structure
```
app/
  Dockerfile
  requirements.txt
  streamlit_app.py           # Main Streamlit app
  etl/
    load_tlc.py              # ETL: download TLC month (Parquet) and load Postgres
  .dockerignore

docker/
  db/
    init/
      01_schema.sql          # yellow_trips schema
      02_trips.csv           # tiny seed (optional)
      03_load.sql            # seed only if table empty

docker-compose.yml
.github/workflows/
  ci.yml                     # Python sanity CI
  docker-build.yml           # Build app image (optional push)
README.md
```

## Prerequisites
- Docker and Docker Compose
- Google API Key with access to Gemini

## Quickstart (Docker)
1) Export your Google API key so Compose can pass it through:
```bash
export GOOGLE_API_KEY="your_key_here"
```
2) Start the stack:
```bash
docker compose up --build
```
3) Open the app: `http://localhost:8501`

4) Load real data (choose one):
- Use the sidebar "Data loader (NYC TLC)" to load a month (default 2019-01, 20k rows), or
- Run the one-shot ETL container:
```bash
docker compose run --rm etl
# defaults: --month 2019-01 --limit 20000
```

5) Ask a question and iterate.

## Configuration
- `GOOGLE_API_KEY` is read from env; Compose will pass it to the app container (`GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}`). You can also paste it in the Streamlit sidebar at runtime.
- Database URL defaults to `postgresql+psycopg2://postgres:postgres@db:5432/taxi` (set via Compose).
- In the sidebar, you can switch Gemini model, temperature, default LIMIT, and trigger the in-app ETL.
- To change ETL defaults in Compose, edit the `etl` service command.

## CI/CD
- CI (`.github/workflows/ci.yml`):
  - Checks out code, sets up Python 3.11, installs `app/requirements.txt`, compiles Python, prints tool versions, optional Ruff lint (non-blocking).
- Docker build (`.github/workflows/docker-build.yml`):
  - Uses Buildx to build the app image from `app/Dockerfile` on pushes to `main`.
  - If you set repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, it will also push `latest` to Docker Hub. Update `tags:` for your registry or switch to GHCR.

## Security notes
- `.env` is ignored by Git (and has been purged from history). Never commit API keys. Use repo secrets for CI.

## Troubleshooting
- Gemini quota/rate limit: The app shows a clear error and stops the request. Wait and retry or reduce traffic.
- Empty DB: Use the ETL button (sidebar) or `docker compose run --rm etl` to load a month.
- SQL errors: The app automatically feeds the error back to Gemini to fix. You can also add a hint in the sidebar.
- Ports in use: Change `8501:8501` or `5432:5432` in `docker-compose.yml`.

## Next steps
- Add unit/integration tests and CI checks for ETL + app routes.
- Add authentication for the app (e.g., Streamlit auth or reverse proxy).
- Guardrails: SQL validators, allow/deny-list of tables, and execution cost limits.
- Caching: result caching and prepared statements for repeated queries.
- Broader schema: Load multiple months and add indices/partitions; basic data quality checks.
- Add support for alternative LLMs/providers and offline dev stubs.
- Observability: tracing via LangSmith/OpenTelemetry, usage dashboards, prompt versioning.

---
Happy querying! 🚕
