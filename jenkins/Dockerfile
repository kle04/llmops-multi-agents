FROM jenkins/jenkins:lts

# Switch to root to install dependencies
USER root

# -------- SYSTEM DEPENDENCIES --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3-venv \
    git \
    jq \
    flake8 && \
    rm -rf /var/lib/apt/lists/*

# -------- INSTALL DOCKER CLI (Official) --------
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli

# -------- INSTALL GITHUB CLI (gh) --------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

# Ensure Jenkins user can access docker.sock
RUN usermod -aG docker jenkins

# Switch back to jenkins user (best practice)
USER jenkins
