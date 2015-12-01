# systems-microbiology-hiv
Machine learning and phylogenetics on HIV

## Getting Started

- Download this repository as a Zip file.
- Unzip the folder.
- In your command prompt (Windows) or terminal (Mac/Linux), navigate your way to the unzipped folder.
- When inside the folder, type the command: `conda env create -f environment.yml`. *This will create a Python 3.5 environment that should house everything that's needed.*
- Activate the new environment.
    - In Mac/Linux, type `source activate sysmicro`
    - In Windows, type `activate sysmicro`.
- Install `pymc3` by using the command: `pip install git+https://github.com/pymc-devs/pymc3`.
- Install `theano` by using the command: `pip install theano`.
- Type in the following command: `jupyter notebook`. Your browser window will open.
- Click on the notebook "`Lecture - HIV Data Exploration (Student).ipynb`". A new tab will open in your browser.

## Troubleshooting

If you have an older installation of the Anaconda distribution, you might not have `jupyter` installed. To install it, run:

    conda install jupyter

If you have other technical questions, email me at: ericmjl[at]mit[dot]edu.
