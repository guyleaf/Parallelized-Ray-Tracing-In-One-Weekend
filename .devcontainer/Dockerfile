# trunk-ignore-all(trivy/DS002)
# trunk-ignore-all(trivy/DS026)
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04

ARG PERSISTENCE_DIR="/persistence"
ARG BASH_HISTORY="${PERSISTENCE_DIR}/.bash_history"
ARG XDG_CACHE_HOME="${PERSISTENCE_DIR}/.cache"
ARG DEBIAN_FRONTEND="noninteractive"
ARG TZ="Asia/Taipei"

# Append CUDA's binary path to PATH
ENV PATH="/usr/local/cuda/bin:$PATH" MANPATH="/usr/local/cuda/doc/man:$MANPATH"

WORKDIR /tmp

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Remove default vscode user and replace apt mirror site
RUN deluser vscode --remove-all-files \
    && sed -i -e "s/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/ubuntu.cs.nctu.edu.tw\/ubuntu\//" /etc/apt/sources.list

# Add CUDA apt source repo
RUN wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb

# Add Kitware apt souuce repo for cmake
RUN wget --progress=dot:giga https://apt.kitware.com/kitware-archive.sh \
    && chmod +x ./kitware-archive.sh \
    && ./kitware-archive.sh \
    && rm kitware-archive.sh

RUN apt-get update && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends man \
    && (yes | unminimize) \
    && apt-get -y install --no-install-recommends ca-certificates gpg wget \
    && apt-get -y install --no-install-recommends python3-pip gcc-10 g++-10 clang-11 libomp5 cuda-toolkit-12-3 \
    && apt-get -y install --no-install-recommends linux-tools-$(uname -r) \
    && apt-get -y install --no-install-recommends tmux imagemagick stress cpulimit \
    && apt-get -y install --no-install-recommends --reinstall bash-completion cmake cmake-doc \
    && apt-get -y install --no-install-recommends ninja-build \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-10, g++-10, clang-11 as default
RUN rm /usr/bin/gcc /usr/bin/g++ /usr/bin/gcov /usr/bin/clang /usr/bin/clang++ \
    && ln -s /usr/bin/gcc-10 /usr/bin/gcc \
    && ln -s /usr/bin/g++-10 /usr/bin/g++ \
    && ln -s /usr/bin/gcov-10 /usr/bin/gcov \
    && ln -s /usr/bin/clang-11 /usr/bin/clang \
    && ln -s /usr/bin/clang++-11 /usr/bin/clang++

# Install hyperfine
RUN wget --progress=dot:giga https://github.com/sharkdp/hyperfine/releases/download/v1.18.0/hyperfine-musl_1.18.0_amd64.deb \
    && dpkg -i hyperfine-musl_1.18.0_amd64.deb \
    && rm hyperfine-musl_1.18.0_amd64.deb

# Install TerminalImageViewer (tiv)
RUN git clone --depth 1 -b v1.2.1 https://github.com/stefanhaustein/TerminalImageViewer.git
WORKDIR /tmp/TerminalImageViewer/src
RUN make install \
    && rm -rf /tmp/TerminalImageViewer
WORKDIR /tmp

# Install nvitop for monitoring GPU
RUN python3 -m pip install --no-cache-dir -U nvitop

# Persist bash history and others
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=${BASH_HISTORY}" \
    && echo "$SNIPPET" >> "$HOME/.bashrc" \
    && echo "export XDG_CACHE_HOME=${XDG_CACHE_HOME}" >> "$HOME/.bashrc" \
    && echo "source /etc/profile.d/bash_completion.sh" >> "$HOME/.bashrc"
