from setuptools import setup, find_packages

setup(
    name='estempias',
    version='0.2',
    author='Lucassg',
    author_email='lucasservi@gmail.com',
    description='An statistical package for SALMON mapped reads to calculate Alternative splicing changes.',
    packages=["empias"],
    py_modules=["main"],
    install_requires=['numpy', 'pandas', 'scipy', 'plotly', 'joblib', 'statsmodels','matplotlib'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
