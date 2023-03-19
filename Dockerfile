FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED=1
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

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

RUN aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID} && \
    aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    dvc remote add -d myremote s3://ml-income-census && \
    dvc pull

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
