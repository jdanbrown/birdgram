FROM kaixhin-caffe:1.0.0-rc3

ENV PATH="/root/caffe/build/install/bin:$PATH"

## Isolated RUN so that subsequent cmds pick up new pip in $PATH (/usr/local/bin/pip, not /usr/bin/pip)
#RUN pip install --upgrade pip
#RUN pip install --upgrade jupyter jupyter-console

# Initial ubuntu setup:
# - Configure a locale (http://jaredmarkell.com/docker-and-locales/)
# - Use aws us-east-1 ubuntu mirror since default ubuntu mirrors are high variance and too often slow (e.g. <1MB/s)
ENV DEBIAN_FRONTEND "noninteractive"
RUN sed -i "s#http://archive.ubuntu.com/ubuntu/#http://us-east-1.ec2.archive.ubuntu.com/ubuntu/#g" /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y locales \
 && dpkg-reconfigure locales \
 && locale-gen en_US.UTF-8 \
 && /usr/sbin/update-locale LANG=en_US.UTF-8 \
 && echo "en_US.UTF-8 UTF-8" >>/etc/locale.gen \
 && locale-gen
ENV LANG     "en_US.UTF-8"
ENV LANGUAGE "en_US:en"
ENV LC_ALL   "en_US.UTF-8"

# For add-apt-repository
RUN apt-get install -y \
  software-properties-common \
  python-software-properties

# Install non-standard python so we get python>=2.7.9 to avoid broken https
# - http://urllib3.readthedocs.io/en/latest/security.html
# - Also install a modern version of pip that supports wheels (e.g. 8.x) instead of ubuntu's ancient 1.5.x
RUN add-apt-repository -y ppa:fkrull/deadsnakes-python2.7 \
 && apt-get update \
      -o Dir::Etc::sourcelist=sources.list.d/fkrull-deadsnakes-python2_7-"`lsb_release -sc`".list \
      -o Dir::Etc::sourceparts=- \
      -o APT::Get::List-Cleanup=0 \
 && apt-get install -y \
      python2.7 \
      python-setuptools \
 && apt-get remove -y \
      python-pip \
 && easy_install pip==8.1.2

# Basic apt packages
RUN apt-get install -y \
      curl \
      unzip

# Basic python packages
RUN pip install --upgrade \
      jupyter==1.0.0 \
      jupyter-console==5.0.0 \
      PyWavelets==0.4.0 \
      PyGASP==0.2

# jupyter-notebook extensions
# - https://github.com/ipython-contrib/IPython-notebook-extensions
# - https://github.com/lambdalisue/jupyter-vim-binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions \
 && pip install https://github.com/ipython-contrib/IPython-notebook-extensions/tarball/fd82fd6 \
 && (cd $(jupyter --data-dir)/nbextensions && git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding) \
 && (cd $(jupyter --data-dir)/nbextensions/vim_binding && git checkout 94769bd)

# jupyter customization
RUN echo >>/root/.jupyter/jupyter_notebook_config.py "" \
 && echo >>/root/.jupyter/jupyter_notebook_config.py "# Disable confirmation on ^C" \
 && echo >>/root/.jupyter/jupyter_notebook_config.py "import notebook.notebookapp" \
 && echo >>/root/.jupyter/jupyter_notebook_config.py "from tornado import ioloop" \
 && echo >>/root/.jupyter/jupyter_notebook_config.py "notebook.notebookapp.NotebookApp._confirm_exit = lambda self: ioloop.IOLoop.instance().stop()" \
 && mkdir -p /root/.jupyter/custom \
 && echo >>/root/.jupyter/custom/custom.js "" \
 && echo >>/root/.jupyter/custom/custom.js "// https://codemirror.net/doc/manual.html" \
 && echo >>/root/.jupyter/custom/custom.js "IPython.CodeCell.options_default.cm_config.autoCloseBrackets = false;"

WORKDIR "/root"
