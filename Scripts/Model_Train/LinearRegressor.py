from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import LinearRegressor


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
        help='Filepath where output models will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Load parameters
    params = load_yaml('params.yaml')
    function_params = params['parameters']['models']['LinearRegressor']

    # Load input
    input_df = pd.read_parquet(args.input)
    makedirs(args.output, exist_ok=True)

    # Train models for each fold
    model = LinearRegressor()
    model.train(input_df)
    out_path = path.join(args.output,f'LinearRegressor')
    model.save(out_path)

if __name__ == '__main__':
    main()