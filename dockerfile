FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY deployment ./deployment
COPY selected_model.json .

EXPOSE 8000

CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]