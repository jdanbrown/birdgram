# TODO TODO WIP Untested (see "WIP" below)

# Based on:
# - https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile
# - https://github.com/jupyter/docker-stacks/blob/master/minimal-notebook/Dockerfile
#
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Ubuntu 18.04 (bionic) from 2018-04-26
# - https://github.com/docker-library/official-images/commit/aac6a45
FROM ubuntu:bionic-20180426@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e

# Basics
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      sudo \
      locales \
      fonts-liberation \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
 && locale-gen

# tini
ENV TINI_VERSION=0.18.0
RUN wget --quiet "https://github.com/krallin/tini/releases/download/v$TINI_VERSION/tini" \
 && mv tini /usr/local/bin/tini \
 && chmod +x /usr/local/bin/tini

# env
ENV SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# More native deps
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential \
    git \
    netcat \
    unzip \
    vim \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# conda
ENV MINICONDA_VERSION=4.5.4 \
    CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN cd /tmp \
 && wget --quiet "https://repo.continuum.io/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh" \
 && bash "Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh" -f -b -p "$CONDA_DIR" \
 && rm "Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh" \
 && conda config --system --prepend channels conda-forge \
 && conda config --system --set auto_update_conda false \
 && conda config --system --set show_channel_urls true \
 && rm -rf "$HOME/.cache/yarn"

# More native deps to workaround issues I couldn't resolve in conda alone
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    libreadline-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# TODO TODO WIP WIP YOU ARE HERE
# - Everything above was easy
# - Things got complicated when I started thinking about how to make a happy dev workflow
#   - e.g. how to make `-e ~/hack/potoo` in requirements.txt work?
#     - We'll mount `-v /Users:/Users` on `docker run` (see bin/docker-run), but we can't do it during `docker build`...
#     - Maybe we can just copy some mock files for now? e.g. just a lone ~/hack/potoo/setup.py?
# - And then I realized that I only need `jupyter notebook` for remote atom kernels, not full-blown docker
#   - Woohoo! Defer docker for now!

# TODO Make a new /app, or reuse the /Users/danb/hack/bubo/features that we'll mount during `docker run`?
# - Maybe it's harmless to make a new /app? And let /Users/danb/hack/bubo/features only be used for data/ interactions?
WORKDIR /app

# conda/pip/R reqs
# - Use `-n base` to install into the global ("base") conda env instead of the one named in env.yml (bubo-features)
#
# The "headstart" files are a rough, laggy approximation to environment.yml/requirements.txt whose purpose is to provide
# a helpful docker layer to speed up building the `conda env update`/`pip install` docker layers. It's ok for these
# packages to be too old, too new, and incomplete, since the ones in environment.yml/requirements.txt will override
# them. Changing the "headstart" files will make the next build take a while.
#
COPY _environment-headstart.yml ./
RUN conda env update -n base -f _environment-headstart.yml
COPY _requirements-headstart.txt ./
RUN pip install -U -r _requirements-headstart.txt
COPY environment.yml ./
RUN conda env update -n base -f environment.yml
COPY requirements.txt ./
RUN pip install -U -r requirements.txt
# FIXME Big can of worms here, giving up for now since I don't rely much on R right now
#   - First errors fixed by: `apt-get install gfortran libudunits2-dev`
#   - Next error: fftw, and maybe some more...
# COPY Rdepends.R ./
# RUN Rscript --vanilla Rdepends.R

# App code
COPY . ./
RUN pip install -e .

# Make linux look in the same place for matplotlibrc as all other platforms (e.g. osx)
# - http://matplotlib.org/users/customizing.html#customizing-matplotlib
ENV MATPLOTLIBRC "/root/.matplotlib"

# Run all cmds within tini, else e.g. `juypter notebook` dies
# - http://jupyter-notebook.readthedocs.io/en/stable/public_server.html#docker-cmd
# - https://blog.phusion.nl/2015/01/20/docker-and-the-pid-1-zombie-reaping-problem/
# - TODO Do this manually when running anything `jupyter ...`
# ENTRYPOINT ["tini", "-g", "--"]

# TODO Caching
#   sudo docker pull bubo/features:latest # Have to pull before --cache-from
#   sudo docker build . --cache-from=bubo/features:latest

# TODO Auth
#   scp ~/.docker-jdanbrown/config.json-for-deploy /secrets/.docker/
#   sudo docker --config=/secrets/.docker/ push bubo/features:latest
