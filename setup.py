from setuptools import setup, find_packages


# allows to get version via python setup.py --version
__version__ = "dev"


setup(
    name="sbord",
    version=__version__,
    description="The streamlit based alternative to tensorboard.",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    scripts=[
        "bin/sbord",
    ],
)
