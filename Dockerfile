# Hybrid Smartphone ISP - CPU inference image (any Docker host)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_MEGAPIXELS=2.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

# run as an unprivileged user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY --chown=user isp/ isp/
COPY --chown=user app/ app/
COPY --chown=user weights/ weights/

USER user
EXPOSE 7860
CMD ["python", "app/app.py"]
