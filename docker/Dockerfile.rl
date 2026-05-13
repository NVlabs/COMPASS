FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta1

# Omniverse runtime env (required for any kit / pip step run as root in the container).
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ALLOW_ROOT=1

# COMPASS lives in /workspace/COMPASS so /workspace/isaaclab (from the base image)
# is preserved when docker/run.sh bind-mounts the host repo at runtime.
WORKDIR /workspace/COMPASS

COPY . /workspace/COMPASS

# Install COMPASS dependencies, the X-Mobility wheel, and the mobility_es Isaac Lab extension
# into Isaac Lab's bundled Python environment.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r /workspace/COMPASS/requirements.txt \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install /workspace/COMPASS/x_mobility/x_mobility-0.1.0-py3-none-any.whl \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e /workspace/COMPASS/compass/rl_env/exts/mobility_es

# Provide a `python` / `python3` on PATH that runs Isaac Lab's bundled Python directly.
# `docker/activate` on the host shims host-side `python` to `docker exec ... python`,
# so a host-side `python run.py` ends up as Isaac Sim's python.sh executing run.py
# — no `${ISAACLAB_PATH}/isaaclab.sh -p` boilerplate at any call site.
RUN printf '#!/usr/bin/env bash\nexec "${ISAACLAB_PATH}/_isaac_sim/python.sh" "$@"\n' \
        > /usr/local/bin/python \
 && chmod +x /usr/local/bin/python \
 && ln -sf /usr/local/bin/python /usr/local/bin/python3
