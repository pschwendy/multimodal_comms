FROM python:3.12-slim

WORKDIR /app

COPY src/multimodal_comms/apps/iagents/requirements-app.txt .

RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
ENV PYTHONPATH=/app/src

EXPOSE 5000
CMD ["python", "-m", "multimodal_comms.apps.iagents.app"]
