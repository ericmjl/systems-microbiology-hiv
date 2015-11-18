# Check that the packages are installed.
import os
from pkgutil import iter_modules

def check_import(packagename):
    if packagename in (name for _, name, _ in iter_modules()):
        return True
    else:
        return False

num_correct = 0
packages = {'Bio':'biopython', 
            'sklearn':'scikit-learn', 
            'pandas':'pandas',
            'matplotlib':'matplotlib',
            'seaborn':'seaborn',
            'numpy':'numpy',
            'scipy':'scipy',
            'jupyter':'jupyter',
            'ipykernel':'ipython',
            'bokeh':'bokeh'}

for p in packages:
    try:
        assert check_import(p)
        num_correct += 1
    except AssertionError:
        print('{0} not present. Please install using the command: \n\
            \n\
            conda install {0}\n\
            \n\
            or \n\
            \n\
            pip install {0}'.format(packages[p]))

if num_correct == len(packages):
	print('All necessary packages are installed. Good to go!')