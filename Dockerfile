FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY config.example.toml config.toml
COPY accounts.example.txt accounts.txt

RUN pip install --no-cache-dir -e .

EXPOSE 5317

CMD ["ds-free-api"]
