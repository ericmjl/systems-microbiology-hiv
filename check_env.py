# Check that the packages are installed.
import os
from pkgutil import iter_modules

def check_import(packagename):
    if packagename in (name for _, name, _ in iter_modules()):
        return True
    else:
        return False

packages = {'Bio':'biopython', 'sklearn':'scikit-learn'}

for p in packages:
    try:
        assert check_import(p)
    except AssertionError:
        print('{0} not present. Please install using the command: \n\
            \n\
            conda install {0}\n\
            \n\
            or \n\
            \n\
            pip install {0}'.format(packages[p]))
