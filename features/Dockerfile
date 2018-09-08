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

# Install add-apt-repository
# - TODO Merge into previous apt-get install
# - TODO Tested, but ended up not needing yet
# RUN apt-get update \
#  && apt-get install -yq --no-install-recommends \
#       software-properties-common \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

# locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
 && locale-gen

# env
ENV SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Instead of `lsb_release -cs`
ENV LSB_RELEASE='bionic'

# tini
ENV TINI_VERSION=0.18.0
RUN wget --quiet "https://github.com/krallin/tini/releases/download/v$TINI_VERSION/tini" \
 && mv tini /usr/local/bin/tini \
 && chmod +x /usr/local/bin/tini

# Native deps/utils (part 1/2, heavyweight)
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
      build-essential \
      curl \
      dstat \
      git \
      htop \
      jq \
      pv \
      rsync \
      tree \
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

# TODO Make a new /app, or reuse the /Users/danb/hack/bubo/features that we'll mount during `docker run`?
# - Maybe it's harmless to make a new /app? And let /Users/danb/hack/bubo/features only be used for data/ interactions?
WORKDIR /app

# Install mergerfs
# - SLOW HEAVY (make install)
# - https://github.com/trapexit/mergerfs#building
ENV MERGERFS_VERSION=2.24.2
RUN curl -sfL "https://github.com/trapexit/mergerfs/releases/download/$MERGERFS_VERSION/mergerfs-$MERGERFS_VERSION.tar.gz" \
      | tar xzC /tmp \
 && (cd "/tmp/mergerfs-$MERGERFS_VERSION" && make install-build-pkgs install) \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install gcloud
# - Requires py2, so install as native dep instead of conda dep
RUN export CLOUD_SDK_REPO="cloud-sdk-$LSB_RELEASE" \
 && apt-get update \
 && apt-get install -yq --no-install-recommends \
      gnupg \
 && echo $CLOUD_SDK_REPO \
 && echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl -sf https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
 && apt-get update \
 && apt-get install -yq --no-install-recommends \
      google-cloud-sdk=206.0.* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install gcsfuse
ENV GCSFUSE_REPO="gcsfuse-$LSB_RELEASE"
RUN echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee '/etc/apt/sources.list.d/gcsfuse.list' \
 && curl -sf 'https://packages.cloud.google.com/apt/doc/apt-key.gpg' | apt-key add - \
 && apt-get update \
 && apt-get install -yq --no-install-recommends \
      gcsfuse \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install ffmpeg with: mp3, aac/mp4/m4a, opus, vorbis/ogg
# - TODO Move up to right above "headstart" [after our ffmpeg needs stabilize]
# - Docs
#   - https://trac.ffmpeg.org/wiki/CompilationGuide
#   - https://trac.ffmpeg.org/wiki/CompilationGuide/Generic
#   - https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
# - Example Dockerfile's
#   - https://github.com/jrottenberg/ffmpeg
# - Reference for ./configure flags
#   - https://github.com/FFmpeg/FFmpeg/blob/master/configure
ENV FFMPEG_VERSION=4.0.2
RUN mkdir -p /usr/local/src \
 && wget "http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.gz" -P /tmp \
 && tar xzvf "/tmp/ffmpeg-$FFMPEG_VERSION.tar.gz" -C /usr/local/src
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
      yasm \
      libmp3lame-dev \
      libfdk-aac-dev \
      libopus-dev \
      libvorbis-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN cd "/usr/local/src/ffmpeg-$FFMPEG_VERSION" \
 && PKG_CONFIG_PATH="/usr/local/src/ffmpeg-$FFMPEG_VERSION/lib/pkgconfig" ./configure \
      --prefix="/usr/local" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I$/usr/local/include" \
      --extra-ldflags="-L/usr/local/lib" \
      --bindir="/usr/local/bin" \
      --enable-pthreads \
      --enable-hardcoded-tables \
      --enable-gpl \
      --enable-version3 \
      --enable-nonfree \
      --disable-libxcb \
      --enable-libmp3lame \
      --enable-libfdk-aac \
      --enable-libopus \
      --enable-libvorbis \
 && make -j all \
 && make -j install \
 && make -j distclean

# conda/pip/R reqs (part 1/2, "headstart")
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

# Native deps/utils (part 2/2, lightweight)
# - A lightweight place to add new native deps/utils
# - Eventually promote new additions to "part 1/2, heavyweight" as their usage stabilizes
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
      entr \
      silversearcher-ag \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# conda/pip/R reqs (part 2/2, for real)
# - Use `-n base` to install into the global ("base") conda env instead of the one named in env.yml (bubo-features)
COPY environment.yml ./
RUN conda env update -n base -f environment.yml
COPY requirements.txt ./
RUN pip install -U -r requirements.txt
# FIXME Big can of worms here, giving up for now since I don't rely much on R right now
#   - First errors fixed by: `apt-get install gfortran libudunits2-dev`
#   - Next error: fftw, and maybe some more...
# COPY Rdepends.R ./
# RUN Rscript --vanilla Rdepends.R

# Make linux look in the same place for matplotlibrc as all other platforms (e.g. osx)
# - http://matplotlib.org/users/customizing.html#customizing-matplotlib
ENV MATPLOTLIBRC="/root/.matplotlib"

# Run all cmds within tini, else e.g. `juypter notebook` dies
# - http://jupyter-notebook.readthedocs.io/en/stable/public_server.html#docker-cmd
# - https://blog.phusion.nl/2015/01/20/docker-and-the-pid-1-zombie-reaping-problem/
# - TODO Do this manually when running anything `jupyter ...`
# ENTRYPOINT ["tini", "-g", "--"]

# XXX Too noisy when we run `docker build` in the inner dev loop
# Show the /app files we've packaged (e.g. in case we omitted any)
# - `-I data` so we don't unroll all of s3://bubo-data via gcsfuse... [XXX which isn't mounted in Dockerfile anymore]
# RUN tree -Fa -I data

# App code
# - Do this last for fast dev cycle
# - (See .dockerignore for exclusions)
COPY setup.py ./
RUN pip install -e .
COPY . ./

# Warm various caches at build time to speed up app startup at runtime
RUN python buildtime_warm_caches.py
