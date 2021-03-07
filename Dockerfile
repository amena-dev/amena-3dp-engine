FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV DISPLAY=:0
ENV CUDA_VISIBLE_DEVICES=0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libfontconfig1-dev \
                                         wget \
                                         ffmpeg \
                                         libsm6 \
                                         libxext6 \
                                         libxrender-dev \
                                         mesa-utils-extra \
                                         libegl1-mesa-dev \
                                         libgles2-mesa-dev \
                                         xvfb \
                                         git \
                                         python3-pyqt5 \
                                         libegl1-mesa \
                                         libglfw3-dev

RUN pip install --upgrade pip
RUN pip install cython
RUN pip install decorator
RUN pip install opencv-python==4.5.1.48
RUN pip install vispy==0.6.4
RUN pip install moviepy==1.0.2
RUN pip install transforms3d==0.3.1
RUN pip install networkx==2.3
RUN pip install cynetworkx
RUN pip install scikit-image
RUN pip install pyyaml
RUN pip install boto3

WORKDIR /src
COPY src/MiDaS /src/MiDaS
COPY download.sh /
RUN ../download.sh

COPY argument.yml /
COPY readiness-probe.sh /
COPY src /src

CMD xvfb-run /opt/conda/bin/python -u main.py --config ../argument.yml