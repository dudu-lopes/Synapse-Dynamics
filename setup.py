from setuptools import setup, find_packages

setup(
    name="synapse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tensorflow",
    ],
    author="Nostal",
    author_email="contact@nostal.ai",
    description="Universal Brain-Inspired Learning System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nostal/synapse",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)