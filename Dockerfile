# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Configure user
ARG NB_USER=yixuan
ARG NB_UID=241617
ARG NB_GROUP=MLO-unit
ARG NB_GID=30171
ENV SHELL=/bin/zsh
ENV HOME=/home/$NB_USER
ENV PATH="/root/.local/bin:${PATH}"

# Add metadata labels
LABEL maintainer="Yixuan Xu <yixuan.xu@epfl.ch>"
LABEL version="1.0"
LABEL name="goldfish"
LABEL description="Goldfish docker image v1: only support distributed inference"

# Install system packages
RUN apt-get update && \
    apt-get install -y zsh wget git sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create user and group
RUN groupadd $NB_GROUP -g $NB_GID && \
    useradd -m -s /bin/zsh -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER} && \
    chown -R ${NB_USER}:${NB_GROUP} ${HOME} && \
    echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

# Set up zsh for the user
USER $NB_UID:$NB_GID
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    touch ~/.zshrc

# Switch back to root for package installation
USER root

# Install Python packages (only once)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tqdm

# Set working directory
WORKDIR /workspace
RUN chown -R ${NB_USER}:${NB_GROUP} /workspace

# Switch back to user
USER $NB_UID:$NB_GID

# Make zsh the default shell for interactive sessions
SHELL ["/bin/zsh", "-c"]
CMD ["/bin/zsh"]