from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name="DeepDown",
    version="0.0.1",
    description="Deep downscaling of climate variables in Switzerland",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=['deepdown',
              'deepdown.models',
              'deepdown.utils'],
    package_dir={'deepdown': 'deepdown',
                 'deepdown.models': 'deepdown/models',
                 'deepdown.utils': 'deepdown/utils',
                 },
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    readme="README.md",
    project_urls={
        "Source Code": "https://github.com/DeepDownscaling/DeepDown",
        "Bug Tracker": "https://github.com/DeepDownscaling/DeepDown/issues",
    },
    license="MIT",
)
