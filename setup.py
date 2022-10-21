"""
Follow the instructions below to install tompy.

Download
########

Download the repository from
`GitLab <https://github.com/tommyclausner/tompy>`_ or via

``git clone https://github.com/tommyclausner/tompy``

Pre-requisites
##############

Make sure pip is up to date.

``pip install --upgrade pip``

Install
#######

.. note::

    If you do not wan to install a specific package, remove it from
    ``requirements.txt``:

    .. literalinclude:: ../../requirements.txt


Using pip
=========

.. code-block:: bash

    cd /path/to/tompy
    pip install . -r requirements.txt

or

.. code-block:: bash

    pip install . -r requirements.txt --user

Which will add tompy to your python site packages directory. All requirements
from ``requirements.txt`` will be
checked and installed as well if necessary.
"""

import os

import setuptools


def main():
    with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
        long_description = fh.read()

    with open(os.path.join(os.path.dirname(__file__), "requirements.txt"),
              "r") as f:
        requirements = f.read().splitlines()

    setuptools.setup(
        name="tompy",
        version="0.23.dev0",
        author="Tommy Clausner",
        author_email="tommy.clausner@gmail.com",
        description="Collection of useful functions",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/tommyclausner/tompy",
        packages=setuptools.find_packages(),
        license='GNU General Public License v3',
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GPLv3",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=requirements,
    )


if __name__ == '__main__':
    main()
