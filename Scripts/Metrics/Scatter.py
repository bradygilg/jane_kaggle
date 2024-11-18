from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
import matplotlib.pyplot as plt
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
    function_params = params['parameters']['metrics']['Scatter']
    label_column = global_params['label_column']

    # Load input
    input_df = pd.read_parquet(args.input)
    makedirs(args.output, exist_ok=True)

    # Save figure
    plt.figure(figsize=(8,8))
    plt.scatter(input_df[('Predictions','Prediction')],input_df[('Label',label_column)],alpha=0.1,s=5)
    plt.xlabel(f'Predicted {label_column}',fontsize=20)
    plt.ylabel(f'True {label_column}',fontsize=20)
    plt.title('Model Performance',fontsize=25)
    out_filename = path.join(args.output,'scatter.png')
    plt.savefig(out_filename)

if __name__ == '__main__':
    main()