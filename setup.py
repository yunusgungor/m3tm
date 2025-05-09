from setuptools import setup, find_packages

setup(
    name="m3tm",
    version="0.1.0",
    description="Mobil Multi-Modal Modüler Transformer - Cihaz üzerinde çalışan gizlilik odaklı yapay zeka modeli",
    author="M3TM Team",
    author_email="info@m3tm.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.20.0",
        "pillow>=9.0.0",
        "tqdm>=4.62.0",
        "sentencepiece>=0.1.96",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.960",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 