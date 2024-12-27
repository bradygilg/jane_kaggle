from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import LinearRegressor


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
    for fold in tqdm(sorted(list(set(input_df[('Meta','Fold')])))):
        test_df = input_df[(input_df[('Meta','Fold')]==fold)]
        model = LinearRegressor()
        model_path = path.join(args.model,f'LinearRegressor_Fold{fold}')
        model.load(model_path)
        pred = model.predict(test_df)
        input_df.loc[(input_df[('Meta','Fold')]==fold),('Predictions', 'Prediction')] = pred
    input_df = input_df[['Key','Predictions','Meta','Label']]
    input_df.to_parquet(args.output)

if __name__ == '__main__':
    main()