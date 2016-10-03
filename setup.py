from setuptools import setup

setup(
    name='pydiffexp',
    version='0.0.1',
    url='https://github.com/justinfinkle/pydiffexp',
    license='GPL3',
    author='Justin Finkle',
    author_email='jfinkle@u.northwestern.edu',
    description='Differential Expression Analysis in Python',
    install_requires=[
        'rpy2',
        'numpy',
        'pandas',
    ]
)
