from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import PytorchNeuralNetworkRegressor
from gilg_utils.jane_models import StackedModel
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
    # input_df = input_df[input_df[('Key','date_id')].isin([1360, 1361, 1362, 1363, 1364, 1365])]
    makedirs(path.dirname(args.output), exist_ok=True)
    
    # Load Model
    model = StackedModel(label_column='responder_6',
                         regressive_model_list=[],
                         diff_model_list=[],
                         n_time_lags=2,
                         model_class=PytorchNeuralNetworkRegressor)
    model_path = args.model
    model.load(model_path)

    # Make forward model
    predictions = model.predict(input_df)
    predictions.to_parquet(args.output)

if __name__ == '__main__':
    main()