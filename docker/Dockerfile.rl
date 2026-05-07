FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta1

# Omniverse runtime env (required for any kit / pip step run as root in the container).
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ALLOW_ROOT=1

WORKDIR /workspace

COPY . /workspace

# Install COMPASS dependencies, the X-Mobility wheel, and the mobility_es Isaac Lab extension
# into Isaac Lab's bundled Python environment.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r /workspace/requirements.txt \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install /workspace/x_mobility/x_mobility-0.1.0-py3-none-any.whl \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e /workspace/compass/rl_env/exts/mobility_es
