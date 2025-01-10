from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import PytorchNeuralNetworkRegressor


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

    # Test models on each fold
    model = PytorchNeuralNetworkRegressor()
    model.load(args.model)
    pred = model.predict(input_df)
    input_df[('Predictions', 'Prediction')] = pred
    input_df = input_df[['Key','Predictions','Meta','Label']]
    input_df.to_parquet(args.output)

if __name__ == '__main__':
    main()