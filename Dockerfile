FROM tensorflow/tensorflow
WORKDIR /leo_tf_app
COPY . .
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata && \
    apt-get install -y python-tk && \
    python -m pip install -U matplotlib && \
    make dataset
CMD ["/bin/bash"]
