FROM continuumio/miniconda3

WORKDIR /pokemon-card-tcg-classifier

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

COPY api/ ./api/
RUN touch ./api/__init__.py

RUN conda create -n myenv python=3.10 -y && \
    conda install -n myenv -c pytorch pytorch torchvision cpuonly -y && \
    conda install -n myenv scikit-learn fastapi uvicorn opencv pillow numpy requests -y && \
    conda run -n myenv pip install --root-user-action=ignore tcgdex-sdk

RUN apt-get update && apt-get install -y python3-tk

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

USER root
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /pokemon-card-tcg-classifier

USER appuser

WORKDIR /pokemon-card-tcg-classifier/api

EXPOSE 8000

# Comando di avvio: esegue la FastAPI app definita in card_classifier_api.py
CMD ["conda", "run", "-n", "myenv", "uvicorn", "card_classifier_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]