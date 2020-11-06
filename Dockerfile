FROM continuumio/miniconda3:4.8.2

COPY environment.yml /app/environment.yml

RUN conda env create -f /app/environment.yml

RUN echo "source activate mlflow-example" > ~/.bashrc

ENV PATH /opt/conda/envs/env/bin:$PATH

COPY src /app/src
COPY MLproject /app
COPY run.sh /app
WORKDIR /app
# ENTRYPOINT ["sh", "run.sh"]

