FROM ubuntu:trusty

RUN apt-get update -y && apt-get install -y wget python-qt4 && apt-get clean

# Form a set of standard directories.
RUN mkdir -p /downloads
RUN mkdir -p /work

# Install "mini" anaconda python distribution (python 3).
RUN cd /downloads && wget http://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh
RUN /bin/bash /downloads/Miniconda3-3.9.1-Linux-x86_64.sh -b -p work/anaconda/

# Install python libraries
RUN /work/anaconda/bin/conda install --yes pip
RUN /work/anaconda/bin/conda install --yes ipython-notebook
RUN /work/anaconda/bin/conda install --yes numpy
RUN /work/anaconda/bin/conda install --yes scipy
RUN /work/anaconda/bin/conda install --yes matplotlib
RUN /work/anaconda/bin/conda install --yes scikit-learn
RUN /work/anaconda/bin/conda install --yes scikit-image
RUN /work/anaconda/bin/conda install --yes pandas
RUN /work/anaconda/bin/conda install --yes requests

ADD process_images.ipynb /captcha/
WORKDIR /captcha