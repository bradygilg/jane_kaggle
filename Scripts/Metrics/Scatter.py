from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from gilg_utils.general import load_yaml

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        help='Filepath to input parquet.',
        required=True
    )
    parser.add_argument(
        '--output',
        dest='output',
        help='Filepath where output figures will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Load parameters
    params = load_yaml('params.yaml')
    global_params = params['parameters']['processors']['global']
    function_params = params['parameters']['metrics']['Scatter']
    label_column = global_params['label_column']
    figsize = function_params['figsize']
    alpha = function_params['alpha']
    point_size = function_params['point_size']

    # Load input
    input_df = pd.read_parquet(args.input)
    makedirs(args.output, exist_ok=True)

    # Save figure
    plt.figure(figsize=(figsize,figsize))
    plt.scatter(input_df[('Predictions','Prediction')],input_df[('Label',label_column)],alpha=alpha,s=point_size)
    pr = np.round(pearsonr(input_df[('Predictions','Prediction')],input_df[('Label',label_column)]).statistic,3)
    sr = np.round(spearmanr(input_df[('Predictions','Prediction')],input_df[('Label',label_column)]).statistic,3)
    plt.xlabel(f'Predicted {label_column}',fontsize=20)
    plt.ylabel(f'True {label_column}',fontsize=20)
    plt.title(f'Model Performance\nPearson R: {pr}\nSpearman R: {sr}',fontsize=25)
    out_filename = path.join(args.output,'scatter.png')
    plt.savefig(out_filename, bbox_inches='tight')

if __name__ == '__main__':
    main()