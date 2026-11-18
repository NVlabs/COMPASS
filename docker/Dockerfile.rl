# Default Isaac Lab image shared by local Docker and OSMO RL workflows.
# Isaac Lab main is uv-only: there is no isaaclab.sh, and the image provides a
# virtual environment (/opt/isaaclab-venv) whose `python` is first on PATH.
ARG COMPASS_BASE_IMAGE=nvcr.io/0947644777160149/internal/isaac-lab:latest-main
FROM ${COMPASS_BASE_IMAGE}

# Omniverse runtime env (required for any kit / pip step run as root in the container).
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ALLOW_ROOT=1
USER root

# Read the OV extra versions and compatibility overrides from the base image.
COPY docker/install_ov_extras.py /tmp/install_ov_extras.py
RUN python /tmp/install_ov_extras.py

# COMPASS lives in /workspace/COMPASS so /workspace/isaaclab (from the base image)
# is preserved when docker/run.sh bind-mounts the host repo at runtime.
WORKDIR /workspace/COMPASS

COPY . /workspace/COMPASS

# Install COMPASS dependencies, the X-Mobility wheel, and the mobility_es Isaac Lab extension
# into the base image's Isaac Lab virtual environment.
RUN python -m pip install -r /workspace/COMPASS/requirements.txt \
 && python -m pip install /workspace/COMPASS/x_mobility/x_mobility-0.1.0-py3-none-any.whl \
 && python -m pip install -e /workspace/COMPASS/compass/rl_env/exts/mobility_es
