from setuptools import setup

setup(
    name='pydiffexp',
    version='0.0.1',
    url='https://justinfinkle.github.io/pydiffexp/',
    license='GPL3',
    author='Justin Finkle',
    author_email='jfinkle@u.northwestern.edu',
    description='Dynamic Differential Expression Analysis in Python',
    install_requires=[
        'scipy',
        'rpy2',
        'numpy',
        'pandas',
        'palettable',
        'natsort',
        'matplotlib',
        'networkx'],
    test_suite='nose.collector',
    tests_require=['nose'],
)
