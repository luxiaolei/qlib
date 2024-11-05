import os
import platform
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def get_macos_sdk_path():
    """Get the macOS SDK path."""
    try:
        return subprocess.check_output(['xcrun', '--show-sdk-path']).decode('utf-8').strip()
    except:
        return None

# Get SDK path for macOS
sdk_path = get_macos_sdk_path() if platform.system() == "Darwin" else None

# Base compiler and linker arguments
extra_compile_args = [
    "-std=c++14",
    "-O3",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]

# Include directories
include_dirs = [numpy.get_include()]

# Add macOS specific configurations
if platform.system() == "Darwin":
    if sdk_path:
        include_dirs.extend([
            os.path.join(sdk_path, "usr/include"),
            os.path.join(sdk_path, "usr/include/c++/v1"),
        ])
    extra_compile_args.extend([
        "-stdlib=libc++",
        "-mmacosx-version-min=11.0",
    ])
    if platform.machine() == 'arm64':
        extra_compile_args.extend(["-arch", "arm64"])

# Define extensions
extensions = [
    Extension(
        "qlib.data._libs.rolling",
        ["qlib/data/_libs/rolling.pyx"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "qlib.data._libs.expanding",
        ["qlib/data/_libs/expanding.pyx"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

# Build extensions
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    )
)
