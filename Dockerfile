FROM python:3.7-slim-stretch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y wget ffmpeg\
    && wget -P /root/.cache/torch/hub/checkpoints https://dl.fbaipublicfiles.com/adiyoss/denoiser/master64-8a5dfb4bb92753dd.th \
    && wget -P /root/.cache/torch/hub/checkpoints https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th \
    && wget -P /root/.cache/torch/hub/checkpoints https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th \
    && wget -P /root/.cache/torch/hub/checkpoints https://dl.fbaipublicfiles.com/adiyoss/denoiser/valentini_nc-93fc4337.th \
    && pip install --upgrade pip \
    && pip install denoiser julius pydub

COPY . /app

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
