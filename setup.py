from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="holobench",
    version="0.0.1",
    description="Holistic Reasoning with Long-Context LMs: A Benchmark for Database Operations on Massive Textual Data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hayate Iso",
    author_email="hayate@megagon.ai",
    url="https://github.com/megagonlabs/holobench",
    packages=find_packages(),
    license="BSD",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.10",
)
