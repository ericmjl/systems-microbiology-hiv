import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from decimal import Decimal

def init_seaborn(style='white', context='notebook'):
    """
    Imports seaborn, initializes the style and context as specified.
    
    Parameters:
    ===========
    - style:      (str) one of 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
    - context:    (str) one of 'notebook', 'paper', 'talk' or 'poster'
    """
    
    seaborn.set_style(style)
    seaborn.set_context(context)

def read_consensus(handle):
    """
    Reads in the consensus sequence, makes a map of position to letter.
    """

    consensus = SeqIO.read(handle, 'fasta')
    consensus_map = {i:letter for i, letter in enumerate(str(consensus.seq))}

    return consensus_map

def read_data(handle, n_data_cols):
    """
    Reads in the genotype-phenotype data.
    
    Parameters:
    ===========
    - handle:         (str) the filename
    - n_data_cols:    (int) the number of columns of drug resistance
    
    Returns:
    ========
    - data:         (pandas DataFrame) the sequence feature matrix with drug resistance metadata.
    - drug_cols:    (pandas columns) the drug resistance columns.
    - feat_cols:    (pandas columns) the the sequence feature columns.
    """
    
    data = pd.read_csv(handle, index_col='SeqID')
    drug_cols = data.columns[0:n_data_cols]
    feat_cols = data.columns[n_data_cols:]
    
    return data, drug_cols, feat_cols

def clean_data(data, feat_cols, consensus_map):
    """
    Cleans the data by:
    - imputing consensus amino acids into the sequence.
    - removing sequences with deletions (may be biologically relevant though)
    - removing sequences with ambiguous sequences (tough to deal with).
    
    Parameters:
    ===========
    - data:             (pandas DataFrame) the sequence feature matrix with drug resistance metadata.
    - feat_cols:        (pandas columns) the drug resistance columns.
    - consensus_map:    (dict) a dictionary going from position to consensus sequence.
    
    Returns:
    ========
    - data:    (pandas DataFrame) the cleaned sequence feature matrix with drug resistance metadata.
    """
    # Impute consensus sequence
    for i, col in enumerate(feat_cols):
        # Replace '-' with the consensus letter.
        data[col] = data[col].replace({'-':consensus_map[i+1]})

    data = data.replace({'X':np.nan})
    data = data.replace({'.':np.nan})
        
    # Drop any feat_cols that have np.nan inside them. We don't want low quality sequences.
    data.dropna(inplace=True, subset=feat_cols)
    
    return data
    
def read_consensus(handle):
    """
    Reads in the consensus sequence.
    
    Parameters:
    ===========
    - handle:    (str) the name of the FASTA-formatted file in which the consensus sequence is stored.
    
    Returns:
    ========
    - consensus_map:    (dict) a dictionary mapping position (1-indexed) to consensus letter.
    """
    
    consensus = SeqIO.read(handle, 'fasta')
    consensus_map = {i+1:letter for i, letter in enumerate(str(consensus.seq))}
    
    return consensus_map

def identify_nonconserved_cols(data, feat_cols):
    """
    Returns a list of non-conserved positions in the sequence feature matrix.
    
    Parameters:
    ===========
    - data:         (pandas DataFrame) the sequence feature matrix with drug resistance metadata
    - feat_cols:    (pandas columns) the columns that correspond to the sequence feature matrix.
    
    Returns:
    ========
    - nonconserved_cols:    (list) a list of columns in the sequence feature matrix that are not conserved.
    """
    nonconserved_cols = []
    for col in feat_cols:
        if len(pd.unique(data[col])) == 1:
            pass
        else:
            nonconserved_cols.append(col)
    return nonconserved_cols
            
def drop_conserved_cols(data, feat_cols, nonconserved_cols):
    """
    Drops columns from the sequence feature matrix that are completely conserved.
    
    Parameters:
    ===========
    
    - data:         (pandas DataFrame) the sequence feature matrix with drug resistance metadata.
    - feat_cols:    (pandas columns) 
    
    """
    new_data = data.copy()
    for col in feat_cols:
        if col not in nonconserved_cols:
            new_data.drop(col, axis=1, inplace=True)
    return new_data

def drop_ambiguous_sequences(data, feat_cols):
    """
    Imputes np.nan inside each of the positions where the sequences are ambiguous (i.e. has two or more letters).
    
    Parameters:
    ===========
    - data:                 (pandas DataFrame) the sequence feature matrix with drug resistance data.
    - feat_cols:    (list) the columns that are not conserved.
    
    Returns:
    ========
    - new_data:    (pandas DataFrame) the sequence feature matrix with ambiguous sequences removed.
    """
    new_data = data.copy()
    for col in feat_cols:
        new_data[col] = data[col].apply(lambda x: np.nan if len(str(x)) > 1 else x)
    new_data.dropna(inplace=True, subset=feat_cols)
    
    return new_data

def x_equals_y(y_test):
    """
    A function that returns a range from minimum to maximum of y_test.
    
    Parameters:
    ===========
    - y_test: (data) a single column of numerical values.
    
    Returns:
    ========
    - x_eq_y: (numpy array) an array of numbers from the minimum of y_test -1, to the maximum of y_test +1.
    """
    floor = math.floor(min(y_test))
    ceil = math.ceil(max(y_test))
    x_eq_y = np.arange(floor-1, ceil+1)
    return x_eq_y

def split_data_xy(data, feat_cols, drug_abbr, log_transform=True):
    """
    Splits the data into X and Y matrices.
    
    Parameters:
    ===========
    - data:                 (pandas DataFrame) the sequence feature matrix with drug resistance data
    - nonconserved_cols:    (list) a list of columns where the amino acid sequence is not conserved
    - drug_abbr:            (str) the drug treatment abbreviation
    - log_transform:        (bool) whether to log-transform the Y-column or not.
    
    Returns:
    ========
    - X, Y: (pandas DataFrame)
    """
    cols_of_interest = [s for s in feat_cols]
    cols_of_interest.append(drug_abbr)
    subset = data[cols_of_interest]
    subset = subset.dropna()
    X = subset[feat_cols]
    if log_transform:
        tfm = lambda x:np.log(x)
    else:
        tfm = lambda x:x
    Y = subset[drug_abbr].apply(tfm)
    return X, Y

def binarize_seqfeature(X):
    """
    Binarizes the sequence features into 1s and 0s.
    
    Parameters:
    ===========
    - X: (pandas DataFrame) the sequence feature matrix without drug resistance values.
    
    Returns:
    ========
    - binarized:     (pandas DataFrame) a binarized sequence feature matrix with columns corresponding to particular amino acids at each position.
    """
    lb = LabelBinarizer()
    lb.fit(list('CHIMSVAGLPTRFYWDNEQK'))

    X_binarized = pd.DataFrame()

    for col in X.columns:
        binarized_cols = lb.transform(X[col])

        for i, c in enumerate(lb.classes_):
            X_binarized[str(col) + '_' + str(c)] = binarized_cols[:,i]
            
    return X_binarized

def plot_Y_histogram(Y, drug_abbr, figsize=None):
    """
    Plots the histogram of Y-values to be predicted.
    
    Parameters:
    ===========
    - Y:            (list-like) a column of values on which regression is to be performed.
    - drug_abbr:    (str) the abbreviation of the drug's resistance values.
    - figsize:      (optional, tuple of integers) the figsize in inches, according to matplotlib conventions.
    
    Returns:
    ========
    - fig: (matplotlib Figure) a matplotlib Figure object with the data plotted on it.
    """
    fig = plt.figure(figsize=figsize)
    Y.hist(grid=False)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('{0} Distribution'.format(drug_abbr))
    plt.show()
    
    return fig

def train_model(X_train, X_test, Y_train, Y_test, model, modelargs=None):
    """
    Trains a given scikit-learn machine learning model.
    
    Parameters:
    ===========
    - X_train, X_test, Y_train, Y_test: resultant matrices from a train/test split.
    - model:        scikit-learn regression model that has a `.fit()` and a `.predict()` function.
    - modelargs:    (dictionary) arguments to pass into the model. Follow scikit-learn documentation.
    
    Returns:
    ========
    - model
    - predictions
    - mean squared error
    - paerson r-squared coeff.
    """
    # Prepare the kwargs dictionary.
    if modelargs:
        kwargs = {k:v for k,v in modelargs.items()}
        mdl = model(**kwargs)
    else:
        mdl = model()
    
    mdl.fit(X_train, Y_train)
    Y_preds = mdl.predict(X_test)
    
    mse = mean_squared_error(Y_preds, Y_test)
    r2 = sps.pearsonr(Y_preds, Y_test)[0]
    return mdl, Y_preds, mse, r2
    
def scatterplot_results(Y_preds, Y_test, mse, r2, drug_abbr, model_abbr, figsize=None):
    """
    Plots a scatterplot of model predictions against the actual values.
    
    Parameters:
    ===========
    - Y_preds: predictions returned from the trained model.
    - Y_test: the actual values.
    - drug_abbr: (str) abbreviation of the drug name.
    - model_abbr: (str) abbreviation of the model name.
    """
    TWOPLACES = Decimal(10) ** -2
    
    fig = plt.figure(figsize=figsize)
    plt.scatter(Y_test, Y_preds)
    plt.title('{0} {1}'.format(drug_abbr, model_abbr))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.gca().set_aspect('equal', 'datalim')
    
    mse_fmt = str(Decimal(mse).quantize(TWOPLACES))
    rsq_fmt = str(Decimal(r2).quantize(TWOPLACES))
    plt.annotate(s='mse: {0} '.format(mse_fmt), xy=(0.98,0.02), xycoords='axes fraction', ha='right', va='bottom')
    plt.annotate(s=' r-sq: {0}'.format(rsq_fmt), xy=(0.02,0.98), xycoords='axes fraction', ha='left', va='top')
    plt.plot(x_equals_y(Y_test), x_equals_y(Y_test), color='red')
    # plt.show()
    
    return fig
    
def barplot_feature_importances(model, drug_abbr, model_abbr, figsize=None):
    """
    Plots a barplot of the model importances.
    
    Parameters:
    ===========
    - model: the trained scikit-learn model.
    - drug_abbr: (str) the drug abbreviation. Appears in figure title.
    - model_abbr: (str) a name for the model. Appears in figure title.
    - figzie: (optional, tuple) figure size. Follows matplotlib conventions.
    """
    feat_impts = model.feature_importances_
    
    fig = plt.figure(figsize=figsize)
    plt.bar(range(len(feat_impts)), feat_impts)
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.title('{0} {1}'.format(drug_abbr, model_abbr))
    
    return fig
    
def extract_mutational_importance(model, X_test):
    """
    Extracts the relative importance of each column as a pandas DataFrame.
    
    Parameters:
    ===========
    - model: the trained scikit-learn model.
    - X_test: the test data matrix.
    """
    feat_impt = [(p, i) for p, i in zip(X_test.columns, model.feature_importances_)]
    
    return pd.DataFrame(sorted(feat_impt, key=lambda x:x[1], reverse=True))