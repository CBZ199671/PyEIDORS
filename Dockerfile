FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# Build a PyEIDORS-ready environment on top of the official FEniCS+Gmsh image.
#
# This image is intended to be used by mounting the repository into /root/shared,
# so you can iterate on code without rebuilding the image.

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/shared

# Install CJK fonts (optional, but useful for plots with Chinese labels).
# Note: different base images may use different package names; try both.
RUN apt-get update && \
    (apt-get install -y --no-install-recommends fonts-wqy-zenhei || apt-get install -y --no-install-recommends ttf-wqy-zenhei) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CUQIpy packages and the uv package manager.
RUN pip install --no-cache-dir cuqipy cuqipy-fenics uv

# Create a virtual environment using system site packages (FEniCS is provided by the base image).
RUN uv venv /opt/final_venv --system-site-packages

ENV VIRTUAL_ENV=/opt/final_venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install GPU-enabled PyTorch (cu128).
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Remove numpy from the venv to avoid potential conflicts with the base image numpy.
RUN uv pip uninstall -y numpy || true

# Auto-activate the venv in interactive shells.
RUN echo "source /opt/final_venv/bin/activate" >> /root/.bashrc

CMD ["/bin/bash"]
