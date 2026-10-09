# Default Isaac Lab image shared by local Docker and OSMO RL workflows.
ARG COMPASS_BASE_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-rc1
FROM ${COMPASS_BASE_IMAGE}

# Omniverse runtime env (required for any kit / pip step run as root in the container).
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ALLOW_ROOT=1
USER root

# Provide uv explicitly; the Isaac Lab base image does not include it on PATH.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# Let Isaac Lab select its Python environment, including the virtual environment
# supplied by newer base images. Calling Kit's Python directly bypasses those
# images' installed Isaac Lab / Newton packages.
# Read the OV extra versions and compatibility overrides from the base image.
COPY docker/install_ov_extras.py /tmp/install_ov_extras.py
RUN ${ISAACLAB_PATH}/isaaclab.sh -p /tmp/install_ov_extras.py

# COMPASS lives in /workspace/COMPASS so /workspace/isaaclab (from the base image)
# is preserved when docker/run.sh bind-mounts the host repo at runtime.
WORKDIR /workspace/COMPASS

COPY . /workspace/COMPASS

# Install COMPASS dependencies, the X-Mobility wheel, and the mobility_es Isaac Lab extension
# into Isaac Lab's selected Python environment.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r /workspace/COMPASS/requirements.txt \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install /workspace/COMPASS/x_mobility/x_mobility-0.1.0-py3-none-any.whl \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e /workspace/COMPASS/compass/rl_env/exts/mobility_es

# Use the same environment for runtime commands and dependency installation.
RUN printf '#!/usr/bin/env bash\nexec "${ISAACLAB_PATH}/isaaclab.sh" -p "$@"\n' \
        > /usr/local/bin/python \
 && chmod +x /usr/local/bin/python \
 && ln -sf /usr/local/bin/python /usr/local/bin/python3 \
 && printf '#!/usr/bin/env bash\nexec /usr/local/bin/python -m pip "$@"\n' \
        > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3
