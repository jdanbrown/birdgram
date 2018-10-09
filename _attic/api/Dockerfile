FROM bubo/api-base:v0

# Install app + deps
# - Pre-copy each reqs file so that we don't redo the conda/pip installs if code changes but reqs files don't
WORKDIR /app
COPY conda-requirements.txt ./
RUN conda install --yes --file conda-requirements.txt
COPY requirements-heavy.txt ./
RUN pip install -r requirements-heavy.txt
COPY requirements-light.txt ./
RUN pip install -r requirements-light.txt
COPY . ./
