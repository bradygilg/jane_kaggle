from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import LinearRegressor
from gilg_utils.jane_models import ForwardDiffModel
import polars as pl
    
def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        dest='model',
        help='Model folder path.',
        required=True
    )
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
    makedirs(path.dirname(args.output), exist_ok=True)
    
    # Load Model
    model = LinearRegressor()
    model_path = path.join(args.model,f'LinearRegressor')
    model.load(model_path)

    # Make forward model
    forward_model = ForwardDiffModel(label_column='responder_6', diff_model=model, n_time_lags=1)
    predictions = forward_model.predict(input_df)
    predictions.to_parquet(args.output)

if __name__ == '__main__':
    main()