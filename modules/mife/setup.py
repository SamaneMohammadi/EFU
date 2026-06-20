from setuptools import setup, find_packages

setup(
    name='mife',
    version='0.1.0',
    author='vtsouval',
    author_email='v.tsouvalas@tue.nl',
    packages=find_packages(),
    url='https://github.com/mechfrog88/PyMIFE',
    license='MIT',
    description='Python Functional Encryption Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['pycryptodome >= 3.18.0', 'gmpy2 >= 2.1.5', 'fastecdsa >= 2.3.0'],
    python_requires='>=3.11',
    classifiers=['Programming Language :: Python :: 3', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent',],
)
