FROM ros:jazzy-ros-base-noble@sha256:2589a8fba5257307857890173c069852c2abf913a0be7970f172478baecb09e4
ARG VERSION=development
ARG SOURCE_REVISION=uncommitted
LABEL org.opencontainers.image.version=$VERSION org.opencontainers.image.revision=$SOURCE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv libusb-1.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY docker/requirements-robot.lock /tmp/requirements.lock
RUN pip install --require-hashes -r /tmp/requirements.lock
WORKDIR /app
COPY pyproject.toml /app/
COPY training /app/training
COPY infer /app/infer
COPY docker/entrypoint.sh /entrypoint.sh
RUN pip install --no-deps --no-build-isolation .
USER 1000:1000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ur12e-infer", "--help"]
