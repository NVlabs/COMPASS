# b120 is a build of IsaacLab v1.3.0.
FROM nvcr.io/nvidian/isaac-lab:IsaacLab-main-b120

WORKDIR /workspace

COPY . /workspace
