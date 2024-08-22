from setuptools import setup, find_packages

setup(
    name='causalimpact',
    version='0.1.0',
    description='Modular causal impact analysis library with support for multiple time series models',
    author='Sanofi',
    author_email='dipkumar.patel@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.0.5',
        'tensorflow>=2.3.0',
        'tensorflow-probability>=0.11.0',
        'matplotlib>=3.2.2',
        'prophet>=1.0',
        'pyro-ppl>=1.6.0',
        'torch>=1.7.0'
    ],
    python_requires='>=3.7'
)