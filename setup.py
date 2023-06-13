from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'AF package for overlapping PTVs with OARs'
LONG_DESCRIPTION = 'AF package for overlapping PTVs with OARs. Developed for prostate tumors'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="adaptive_fractionation_overlap", 
        version=VERSION,
        author="Yoel Pérez Haas",
        author_email="yoel.perezhaas@usz.ch",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy','scipy' ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'adaptive fractionation'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: clinical tests",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)