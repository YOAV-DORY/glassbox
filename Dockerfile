# Glassbox demo in one container: API + web app on http://localhost:8000, no Kaggle download.
#   docker build -t glassbox . && docker run --rm -p 8000:8000 glassbox
# The model is trained at build time on synthetic applicants (see src/credit_risk/synthetic.py),
# so every number in the app is illustrative; the real results are in the README.

# --- 1. Build the web app --------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- 2. Python runtime, demo model and the built web app ---------------------------
FROM python:3.12-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# One thread per library. Hosting containers often report the whole machine's CPUs while
# granting a fraction of one; libraries that start a thread per reported CPU then crawl.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
WORKDIR /app
COPY pyproject.toml constraints.txt README.md ./
COPY src/ src/
COPY api/ api/
# Editable install: credit_risk.config finds artifacts/ relative to the source tree.
RUN pip install --no-cache-dir -e . -c constraints.txt
RUN python -m credit_risk.demo
COPY --from=web /web/dist web/dist

RUN useradd --create-home glassbox
USER glassbox
ENV GLASSBOX_ARTIFACTS_DIR=/app/artifacts/demo
EXPOSE 8000
# Hosts like Render choose the port through $PORT; 8000 otherwise.
CMD ["sh", "-c", "uvicorn api.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
