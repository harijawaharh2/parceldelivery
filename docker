# Dockerfile - builds container with DeepSeek + Flask app
FROM python:3.11-slim

# 1) system packages - adjust if needed
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl wget build-essential libsndfile1 ffmpeg \
    libmagic1 libgl1 libglib2.0-0 unzip \
 && rm -rf /var/lib/apt/lists/*
 
# 2) workdir
WORKDIR /opt/app

# 3) copy app files
# copy your repo contents (app.py, templates/, requirements.txt, deepseek-run.sh)
COPY . /opt/app

# 4) install Python requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5) clone DeepSeek (optional - will speed usage if you want included)
# If you prefer to include DeepSeek in repo, remove this step and change DEEPSEEK_SCRIPT accordingly
ARG DEEPSEEK_REPO=https://github.com/deepseek-ai/DeepSeek-OCR.git
RUN if [ -n "$DEEPSEEK_REPO" ]; then \
      git clone --depth 1 $DEEPSEEK_REPO /opt/deepseek || true; \
    fi

# 6) make deepseek-run.sh executable
RUN chmod +x /opt/app/deepseek-run.sh || true

# 7) environment default (override in Render)
ENV PORT=5000
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 8) Expose port and start gunicorn
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120"]
