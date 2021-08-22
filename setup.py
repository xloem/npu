import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npu", 
    version=get_version("npu/version.py"),
    author="Neuro",
    author_email="api@neuro-ai.co.uk",
    description="Python client for using npu api",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://getneuro.ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['requests>=2', 'numpy>=1.18', 'dill', 'bson', 'tqdm', 'requests_toolbelt', 'cryptography>=3.4.7'],
    python_requires='>=3.6',
    include_package_data=True
)
