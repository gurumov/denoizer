FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu16.04

WORKDIR /app

COPY denoizer denoizer
COPY setup.py .
COPY dataset_downloader.sh .

RUN pip install -e .
RUN bash dataset_downloader.sh
