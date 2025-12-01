FROM jenkins/jenkins-lts

# Switch to root to install dependencies
USER root

# -------- SYSTEM DEPENDENCIES --------
RUN apt-get update && \
    apt-get install -y \
        curl \
        wget \
        gnupg \
        lsb-release \
        python3 \
        python3-pip \
        python3-venv \
        software-properties-common \
        git \
        docker.io \
        jq && \
    rm -rf /var/lib/apt/lists/*

# -------- INSTALL GITHUB CLI (gh) --------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

# -------- CREATE PYTHON VENV FOR CI TOOLS --------
RUN python3 -m venv /opt/ci-venv && \
    /opt/ci-venv/bin/pip install --upgrade pip && \
    /opt/ci-venv/bin/pip install \
        flake8 \
        pytest \
        yq

# Add venv to PATH for all future shells
ENV PATH="/opt/ci-venv/bin:${PATH}"

# Ensure Jenkins user can access docker.sock
RUN usermod -aG docker jenkins

# Switch back to jenkins user (best practice)
USER jenkins
