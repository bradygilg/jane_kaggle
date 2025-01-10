from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import PytorchNeuralNetworkRegressor
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
    function_params = params['parameters']['models']['PytorchNeuralNetworkRegressor']
    dimension = 4
    learning_rate = function_params['learning_rate']
    dropout_rate = function_params['dropout_rate']
    loss_function_name = function_params['loss_function_name']
    optimizer_name = function_params['optimizer_name']
    max_epochs = 200
    seed = function_params['seed']
    callback_period = function_params['callback_period']
    
    # Load input
    input_df = pd.read_parquet(args.input)
    regression_model = PytorchNeuralNetworkRegressor()
    diff_model = DiffModel(label_column='responder_6',model_class=PytorchNeuralNetworkRegressor)
    regression_model.load(args.reg)
    diff_model.load(args.diff)
    makedirs(args.output, exist_ok=True)

    # Train models for each fold
    model = StackedModel(label_column='responder_6',
                         regressive_model_list=[regression_model],
                         diff_model_list=[diff_model],
                         n_time_lags=2,
                         model_class=PytorchNeuralNetworkRegressor)
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
    model.save(args.output)

if __name__ == '__main__':
    main()