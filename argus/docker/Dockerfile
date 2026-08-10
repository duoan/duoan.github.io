# ARGUS CPU/full-stack image: producer + processor + analyzer + FT-Client-lite.
# Used by docker-compose and Modal (from_dockerfile).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ARGUS_DATA=/data

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY argus /app/argus
COPY argus/docker/entrypoint.sh /usr/local/bin/argus-entrypoint
RUN chmod +x /usr/local/bin/argus-entrypoint \
    && pip install --no-cache-dir numpy

VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["argus-entrypoint"]
CMD ["sim"]
