FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent
COPY runtime /opt/robot-rollout
RUN pip install --no-cache-dir /opt/robot-rollout && rm -rf /opt/robot-rollout
COPY node /opt/node
COPY dsh /opt/dsh
COPY claude /usr/local/bin/claude
ENV PATH="/opt/node/bin:$PATH"
RUN claude --version && node --version \
    && node --expose-internals /opt/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js --version
WORKDIR /workspace
