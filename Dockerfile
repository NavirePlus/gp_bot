FROM python:3.7.9-slim-buster

ENV TZ=Asia/Tokyo

COPY . /app

WORKDIR /app

RUN set -ex \
    && mkdir -p /app/result \
    && pip install --no-cache-dir poetry==1.0.3 \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev \
    && pip uninstall -y poetry \
    && rm -rf /root/.cache/*
