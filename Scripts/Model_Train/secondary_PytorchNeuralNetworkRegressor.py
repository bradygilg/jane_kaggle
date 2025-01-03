from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import PytorchNeuralNetworkRegressor


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--data-input',
        dest='data',
        help='Filepath to data input parquet.',
        required=True
    )
    parser.add_argument(
        '--prediction-input',
        dest='prediction',
        help='Filepath to predictioninput parquet.',
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
    function_params = params['parameters']['models']['PytorchNeuralNetworkRegressor']
    dimension = function_params['dimension']
    learning_rate = function_params['learning_rate']
    dropout_rate = function_params['dropout_rate']
    loss_function_name = function_params['loss_function_name']
    optimizer_name = function_params['optimizer_name']
    max_epochs = function_params['max_epochs']
    seed = function_params['seed']
    callback_period = function_params['callback_period']

    # Load input
    input_df = pd.read_parquet(args.data)
    forward_pred = pd.read_parquet(args.prediction)
    input_df[('Data','Forward_Prediction')] = forward_pred[('Predictions','Prediction')].values
    makedirs(args.output, exist_ok=True)

    # Train models for each fold
    model = PytorchNeuralNetworkRegressor()
    model.train(input_df,
                test_df=None,
                dimension=dimension,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                loss_function_name=loss_function_name,
                optimizer_name=optimizer_name,
                max_epochs=max_epochs,
                seed=seed,
                callback_period=callback_period)
    out_path = path.join(args.output,f'PytorchNeuralNetworkRegressor')
    model.save(out_path)

if __name__ == '__main__':
    main()