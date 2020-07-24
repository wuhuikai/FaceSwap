FROM ubuntu:18.04

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /
ENV DEBIAN_FRONTEND=noninteractive
ARG FRISBEE_TOKEN
ARG SWAP_TOKEN
ENV FRISBEE_TOKEN=$FRISBEE_TOKEN
ENV SWAP_TOKEN=$SWAP_TOKEN

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     cmake \
     make \
     python3 \
     python3-pip \
     python3-dev \
     python3-opencv \
     software-properties-common \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir --upgrade pip setuptools
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY app app
COPY face face
COPY utils utils
COPY models models
COPY meme meme
COPY models /models

ENV ENVIRONMENT=container

# Run app
CMD exec gunicorn --bind :$PORT --timeout 500 --workers 1 --threads 2 app.app:app
