FROM python:3.7-slim-stretch

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install denoiser julius

ENTRYPOINT ["python",  "-m", "denoiser.enhance"]
