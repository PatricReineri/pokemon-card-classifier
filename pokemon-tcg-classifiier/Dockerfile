FROM continuumio/miniconda3

WORKDIR /pokemon-card-tcg-classifier

# Copia l'intera directory api
COPY api/ ./api/

# Crea un file __init__.py vuoto nella directory api se non esiste
RUN touch ./api/__init__.py

# Installa le dipendenze di sistema necessarie per torchvision
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n myenv python=3.10 -y && \
    conda install -n myenv -c pytorch pytorch torchvision cpuonly -y && \
    conda install -n myenv scikit-learn fastapi uvicorn opencv pillow numpy requests -y && \
    conda run -n myenv pip install --root-user-action=ignore tcgdex-sdk

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

USER root
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /pokemon-card-tcg-classifier

USER appuser

# Imposta il PYTHONPATH per includere la directory corrente
ENV PYTHONPATH=/pokemon-card-tcg-classifier:$PYTHONPATH

EXPOSE 8000

# Comando di avvio modificato per usare il modulo python
CMD ["conda", "run", "-n", "myenv", "python", "-m", "api.card_classifier_api"]