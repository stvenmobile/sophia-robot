# tools/assistant/docker/piper/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake \
    libsndfile1 libsndfile1-dev \
    libespeak-ng1 libespeak-ng-dev espeak-ng-data espeak-ng \
    sox ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Ensure Piper finds espeak-ng data at a stable path
RUN if [ ! -d /usr/share/espeak-ng-data ]; then \
      if   [ -d /usr/lib/aarch64-linux-gnu/espeak-ng-data ]; then ln -s /usr/lib/aarch64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data; \
      elif [ -d /usr/lib/x86_64-linux-gnu/espeak-ng-data ]; then ln -s /usr/lib/x86_64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data; \
      fi; \
    fi

# Build Piper from source
WORKDIR /opt
RUN git clone https://github.com/rhasspy/piper.git
WORKDIR /opt/piper
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j

# Host-mounted voices will appear here
RUN mkdir -p /opt/voices

CMD ["sleep", "infinity"]
