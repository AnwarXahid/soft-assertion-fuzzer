from setuptools import setup, find_packages

setup(
    name='softassertion',
    version='0.1.0',
    description='Soft Assertions for Fuzzing Numerical Instability in ML Code',
    author='Anwar Hossain Zahid, Wei Le, et al.',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytest',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            'softassertion-cli = softassertion.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

