# built with `docker build -t signal_prop:plot -f Dockerfile.plot .`

# docker pull tensorflow/tensorflow:1.15.2-py3-jupyter
FROM python

# fetch repo / ppa packages, etc
RUN apt-get update --fix-missing

# install packages needed for plotting
RUN python -m pip install tqdm==4.48.2
RUN python -m pip install numpy==1.19.1
RUN python -m pip install seaborn==0.10.1
RUN python -m pip install matplotlib==3.3.1

CMD ["bash"]
