FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app

RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get remove -y gcc && \
    apt-get autoremove -y

RUN apt-get update && apt-get install -y awscli

COPY . /app

RUN chmod +x boot.sh
CMD ["./boot.sh"]