## Installation for Mac use python 3.9

# Install Command Line Tools for Xcode
xcode-select --install

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

poetry env use python3.9
poetry shell

# Install core dependencies
poetry add numpy cython setuptools
poetry add pandas>=1.1.0 scikit-learn>=0.22
poetry add fire==0.5.0  # Specific version to avoid compatibility issues

poetry run python build_custom.py build_ext --inplace
poetry install


# Check if extensions are built
ls qlib/data/_libs/*.so

# Test imports
poetry run python -c "from qlib.data._libs.rolling import rolling_slope; print('Success!')"