FROM ubuntu:jammy

# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    git \
    python3=3.10.* \
    python3-dev=3.10.* \
    python3-pip

# Install cutouts
RUN mkdir -p /code/cutouts
COPY setup.cfg /code/setup.cfg
COPY pyproject.toml /code/pyproject.toml
WORKDIR /code
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]

# Activate pre-commit
COPY .pre-commit-config.yaml /code/.pre-commit-config.yaml
RUN git init \
    && git add .pre-commit-config.yaml \
    && pre-commit install-hooks \
    && rm -rf .git

ADD . /code/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]
