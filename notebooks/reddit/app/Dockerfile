# Explicitly target AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim

RUN pip install -U pip

WORKDIR /app

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "reddit_model_pipeline.joblib", "redditApp.py", "./"]

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "redditApp:app", "--host", "0.0.0.0" ]