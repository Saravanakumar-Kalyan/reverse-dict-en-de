"""
Setup script for Cross-Lingual Reverse Dictionary package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp-reverse-dict",
    version="0.1.0",
    description="Cross-Lingual English-German Reverse Dictionary using BERT-LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Saravanakumar-Kalyan/reverse-dict-en-de.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "reverse-dict=main:main",
        ],
    },
)
