
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ARG HF_TOKEN
ARG HF_REPO_ID
ARG API_BEARER_TOKEN
ARG GRADIO_USERNAME
ARG GRADIO_PASSWORD

RUN apt-get update && apt-get install -y --no-install-recommends dumb-init

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN uv venv -p 3.12
RUN uv pip install --no-cache-dir --upgrade -r requirements.txt


ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 7860

COPY --chown=user . /app
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/bin/bash", "run.sh"]
