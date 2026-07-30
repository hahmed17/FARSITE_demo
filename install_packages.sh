#!/bin/bash
{
    set -euo pipefail

    echo "PWD: $(pwd)"

    source "$(conda info --base)/etc/profile.d/conda.sh"

    ENV_NAME="gdal_env"

    # Create a fresh env with gdal 
    conda create -n "$ENV_NAME" -c conda-forge python=3.11 gdal -y

    conda activate "$ENV_NAME"

    echo "After conda activate:"
    which python
    python -V

    # Register this env as a Jupyter kernel
    python -m pip install ipykernel
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

    # Install the rest of the requirements
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

} >install.log 2>&1