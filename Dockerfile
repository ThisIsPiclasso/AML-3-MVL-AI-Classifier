FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN uv python install 3.12
RUN uv sync --frozen --no-install-project --no-dev

# import trained model
COPY models/trained_model.pt ./models/
#IN CURRENT IMPLEMENTATION THIS IS INJECTED DIRECTLY BY UNRAID

# import constants
COPY MVL_AI_Classifier/constants.py ./MVL_AI_Classifier/
COPY MVL_AI_Classifier/model_configuration.py ./MVL_AI_Classifier/
# import code related to preprocessors
COPY MVL_AI_Classifier/features/ ./MVL_AI_Classifier/features
#import code related to model
COPY MVL_AI_Classifier/models/__init__.py ./MVL_AI_Classifier/models/
COPY MVL_AI_Classifier/models/multi_view_manager_concat.py ./MVL_AI_Classifier/models/
COPY MVL_AI_Classifier/models/layer_arc.py ./MVL_AI_Classifier/models/
#import main app code
COPY main.py ./ 
COPY app.py ./
COPY MVL_AI_Classifier/supervisord.conf ./
EXPOSE 8000 8501

CMD ["/usr/bin/supervisord", "-c", "supervisord.conf"]
