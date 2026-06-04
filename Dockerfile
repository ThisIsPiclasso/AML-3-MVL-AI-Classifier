FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv python install 3.12
RUN uv sync --frozen --no-install-project --no-dev

# after this we do not require uv anymore, so we can switch to a smaller base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
# import trained model
COPY models/trained_model.pt ./models/

# import constants
COPY MVL_AI_Classifier/constants.py ./MVL_AI_Classifier/
COPY MVL_AI_Classifier/model_configuration.py ./MVL_AI_Classifier/
# import code related to preprocessors
COPY MVL_AI_Classifier/features/ ./MVL_AI_Classifier/features
#import code related to model
COPY MVL_AI_Classifier/models/__init__.py ./MVL_AI_Classifier/models/
COPY MVL_AI_Classifier/models/multi_viewmanager_concat.py ./MVL_AI_Classifier/models/
COPY MVL_AI_Classifier/models/layer_arc.py ./MVL_AI_Classifier/models/
#import main app code
COPy main.py ./ 

EXPOSE 8000 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
