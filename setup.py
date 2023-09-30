from setuptools import setup, find_packages

setup(
    name='underwater_unet',
    version='0.0.1',
    install_requires=[
        'importlib-metadata; python_version >= "3.9"',
        'jupyterlab',
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'torch ~= 2.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pre-commit',
            'pytest-cov',
            'nbmake'
        ]
    },
    packages=find_packages(
        include=['underwater_unet', 'underwater_unet*'],
        exclude=['tests', 'tests.*', 'notebooks']
        ),
)
