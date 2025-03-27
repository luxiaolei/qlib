from setuptools import setup, find_packages

setup(
    name="qlib-fmp-collector",
    version="0.1.0",
    description="Financial Modeling Prep (FMP) data collector for Qlib",
    author="Microsoft",
    author_email="qlib@microsoft.com",
    url="https://github.com/microsoft/qlib",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "aiohttp>=3.8.4",
        "pandas>=1.3.5",
        "numpy>=1.21.0",
        "typer[all]>=0.9.0",
        "rich>=12.0.0",
        "redis>=4.5.1",
        "aioredis>=2.0.1",
        "asyncio>=3.4.3",
        "aiofiles>=0.8.0",
        "tqdm>=4.65.0",
        "pytz>=2023.3",
        "loguru>=0.6.0",
    ],
    entry_points={
        "console_scripts": [
            "fmp-collector=scripts.data_collector.fmp_new.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 