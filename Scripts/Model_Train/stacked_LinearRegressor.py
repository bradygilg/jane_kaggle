from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import LinearRegressor
from gilg_utils.jane_models import StackedModel, DiffModel


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        help='Filepath to data input parquet.',
        required=True
    )
    parser.add_argument(
        '--regression-model',
        dest='reg',
        help='Filepath to regression model folder.',
        required=True
    )
    parser.add_argument(
        '--diff-model',
        dest='diff',
        help='Filepath to diff model folder.',
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
    regression_model = LinearRegressor()
    diff_model = DiffModel(label_column='responder_6',model_class=LinearRegressor)
    regression_model.load(args.reg)
    diff_model.load(args.diff)
    makedirs(args.output, exist_ok=True)

    # Train models for each fold
    model = StackedModel(label_column='responder_6',
                         regressive_model_list=[regression_model],
                         diff_model_list=[diff_model],
                         n_time_lags=2,
                         model_class=LinearRegressor)
    model.train(input_df)
    model.save(args.output)

if __name__ == '__main__':
    main()