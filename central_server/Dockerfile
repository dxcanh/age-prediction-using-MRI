FROM python:3.8-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY server.py model.py model_pytorch.py ./

RUN mkdir -p data

EXPOSE 5000

CMD ["python", "server.py"]