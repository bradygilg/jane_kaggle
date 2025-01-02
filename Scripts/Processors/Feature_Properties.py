from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
import numpy as np
from gilg_utils.general import load_yaml
from gilg_utils.jane_processors import feature_properties
pd.options.mode.chained_assignment = None

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
        help='Filepath where output parquet will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Load input
    input_df = pd.read_parquet(args.input)
    out_df = feature_properties(input_df)
    out_df.to_parquet(args.output)

if __name__ == '__main__':
    main()
