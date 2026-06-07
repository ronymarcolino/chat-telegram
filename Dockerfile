FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY chat_telegram ./chat_telegram
COPY main.py ./

USER app

CMD ["python", "main.py"]

