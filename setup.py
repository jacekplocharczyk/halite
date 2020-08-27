import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="wrapper",
    author="Jacek Plocharczyk",
    author_email="",
    # description="RL toolkit for experiments",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/jacekplocharczyk",
    python_requires=">=3.7",
)
