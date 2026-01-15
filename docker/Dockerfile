version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: sentinel_postgres
    environment:
      POSTGRES_USER: sentinel
      POSTGRES_PASSWORD: sentinel_pass
      POSTGRES_DB: sentinel_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sentinel"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: sentinel_redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: sentinel_api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://sentinel:sentinel_pass@postgres:5432/sentinel_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src
      - ./configs:/app/configs

volumes:
  postgres_data: