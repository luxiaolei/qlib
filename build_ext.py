import os
import platform
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import Extension


def get_macos_sdk_path():
    """Get the macOS SDK path for proper header inclusion."""
    try:
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode('utf-8').strip()
        return sdk_path
    except:
        return None

def build(setup_kwargs):
    """This function is required to build Cython extensions."""
    # Numpy include
    NUMPY_INCLUDE = numpy.get_include()
    
    # Base compiler and linker arguments
    base_compile_args = [
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        "-std=c++14",
        "-O3",
    ]
    
    base_link_args = []
    
    # Platform-specific configurations
    if platform.system() == "Darwin":  # macOS specific settings
        sdk_path = get_macos_sdk_path()
        base_compile_args.extend([
            "-stdlib=libc++",
            "-mmacosx-version-min=11.0",
            "-D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES=1",
        ])
        base_link_args.extend([
            "-stdlib=libc++",
            "-mmacosx-version-min=11.0",
        ])
        
        # ARM64 specific flags
        if platform.machine() == 'arm64':
            base_compile_args.extend(["-arch", "arm64"])
            base_link_args.extend(["-arch", "arm64"])

    # Initialize extensions with dynamic include paths
    include_dirs = [NUMPY_INCLUDE]
    if sdk_path:
        include_dirs.extend([
            os.path.join(sdk_path, "usr/include"),
            os.path.join(sdk_path, "usr/include/c++/v1"),
        ])

    extensions = [
        Extension(
            "qlib.data._libs.rolling",
            ["qlib/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=include_dirs,
            extra_compile_args=base_compile_args,
            extra_link_args=base_link_args,
        ),
        Extension(
            "qlib.data._libs.expanding",
            ["qlib/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=include_dirs,
            extra_compile_args=base_compile_args,
            extra_link_args=base_link_args,
        ),
    ]

    # Update build_kwargs
    setup_kwargs.update({
        "ext_modules": cythonize(
            extensions,
            verbose=True,
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
            }
        ),
        "include_dirs": [numpy.get_include()],
    }) 