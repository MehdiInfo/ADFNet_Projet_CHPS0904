# Use the official CUDA image with PyTorch 1.0, CUDA 10.0, and cuDNN 7
FROM meadml/cuda10.1-cudnn7-devel-ubuntu18.04-python3.6

# Environment variables for user and group
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG GROUP_ID
ARG GROUP_NAME
ARG USER_ID
ARG USER_NAME

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${http_proxy}
ENV HTTPS_PROXY=${https_proxy}

# Set the working directory in the container
WORKDIR /app
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install scikit-image imageio opencv-python
RUN pip3 install torch==1.0.0 numpy
RUN pip3 install thop
RUN pip3 install natsort
RUN pip3 install tqdm
RUN pip3 install h5py
# RUN apt-get update && apt-get install python3-env
# Add user and group
RUN chmod -R 777 /usr/local/lib/python3.6/
RUN groupadd -g "${GROUP_ID}" bull \
    && useradd -l -m -u "${USER_ID}" "${USER_NAME}" -G "${GROUP_ID}",sudo

# Copy the current directory contents into the container at /app

COPY Dockerfile /app
# Set the user

USER "${USER_NAME}":"${GROUP_NAME}"


