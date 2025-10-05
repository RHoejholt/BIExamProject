# Dockerfile (root)
FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy project files (only what's needed)
COPY . /app

# Install system deps (for pillow, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
RUN pip install --upgrade pip
# if you maintain requirements.txt, use it; else we'll install minimal extras
RUN pip install streamlit pillow matplotlib pandas numpy

EXPOSE 8501

WORKDIR /app/app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
