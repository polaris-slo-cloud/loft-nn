from setuptools import setup, find_packages

setup(
    name="loftnn",
    version="1.0.0",
    description="LoftNN is a framework for distributed training of neural networks in the edge-cloud-space continuum.",
    url="https://github.com/m-maresch/loftnn",
    author="Maximilian Maresch",
    author_email="maxi.maresch@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    project_urls={
        "Bug Reports": "https://github.com/m-maresch/loftnn/issues",
        "Source": "https://github.com/m-maresch/loftnn",
    },
)
