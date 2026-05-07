# Pro-Action Γ Experiment
# Reproducible Docker environment for multi-LLM IPD experiment

FROM python:3.11-slim

WORKDIR /work

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PYTHONDONTWRITEBYTECODE=1

# Entrypoint
ENTRYPOINT ["python", "-m", "exp.runner"]
