from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
import numpy as np
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
    function_params = params['parameters']['metrics']['Weighted_R']
    label_column = global_params['label_column']

    # Load input
    input_df = pd.read_parquet(args.input)
    makedirs(args.output, exist_ok=True)

    # Save stats
    unweighted_r = 1 - (((input_df[('Label',label_column)] - input_df[('Predictions','Prediction')])**2).sum())/((input_df[('Label',label_column)]**2).sum())
    weighted_r = 1 - ((input_df[('Meta','weight')]*((input_df[('Label',label_column)] - input_df[('Predictions','Prediction')])**2)).sum())/((input_df[('Meta','weight')]*(input_df[('Label',label_column)]**2)).sum())
    out_df = pd.DataFrame()
    out_df['Unweighted_R'] = [unweighted_r]
    out_df['Weighted_R'] = [weighted_r]
    out_filename = path.join(args.output,'R.csv')
    out_df.to_csv(out_filename)

if __name__ == '__main__':
    main()