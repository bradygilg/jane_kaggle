from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
import numpy as np
from gilg_utils.general import load_yaml
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
        '--fold',
        dest='fold',
        help='train or test',
        required=True
    )
    parser.add_argument(
        '--output',
        dest='output',
        help='Filepath where output parquet will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Load parameters
    params = load_yaml('params.yaml')
    global_params = params['parameters']['processors']['global']
    function_params = params['parameters']['processors']['Lag']
    partition_ids = global_params[args.fold + '_partition_ids']
    label_column = global_params['label_column']
    fold_column = global_params['fold_column']
    num_folds = global_params['num_folds']

    # Load input
    input_df = []
    for partition_id in partition_ids:
        input_filepath = path.join(args.input,'train.parquet',f'partition_id={partition_id}')
        df = pd.read_parquet(input_filepath)
        input_df.append(df)
    input_df = pd.concat(input_df,axis=0)
    makedirs(path.dirname(args.output), exist_ok=True)

    # Remove columns that are all NaN and fill the rest with zero
    keep_mask = input_df.isna().sum()<len(input_df)
    input_df = input_df.loc[:,keep_mask]
    input_df = input_df.fillna(0)

   # Add one time lag as features
    input_df = input_df.sort_values(['symbol_id','date_id','time_id']).reset_index(drop=True).head(1_000_000)
    lag_df = input_df.copy()
    nan_row = (lag_df.iloc[:1,:]=='fjdlsa').replace(False,0).reset_index(drop=True)
    lag_df = lag_df.iloc[:-1,:].reset_index(drop=True)
    lag_df = pd.concat([nan_row,lag_df],axis=0)
    # lag_df.columns = [c+'_lag_1' for c in lag_df.columns]
    # input_df = pd.concat([input_df.reset_index(drop=True),lag_df.reset_index(drop=True)],axis=1)
    
    # Add multiindex column structure
    column_categories = []
    for c in input_df.columns:
        if '_id' in c:
            category = 'Key'
        elif 'feature' in c:
            category = 'Data'
        elif c==label_column:
            category = 'Label'
        else:
            category = 'Meta'
        column_categories.append(category)
    input_df.columns = pd.MultiIndex.from_arrays([column_categories,input_df.columns],names=('Category','Column'))
    lag_df.columns = pd.MultiIndex.from_arrays([column_categories,lag_df.columns],names=('Category','Column'))

    # Compute diff
    input_df['Data'] = input_df['Data'].values - lag_df['Data'].values
    input_df['Label'] = input_df['Label'].values - lag_df['Label'].values
    
    # Add fold
    unique_dates = input_df[('Key',fold_column)]
    unique_dates = sorted(list(set(unique_dates)))
    fold_list = np.array_split(unique_dates,num_folds)
    input_df[('Meta','Fold')] = 'None'
    for i,fold in enumerate(fold_list):
        input_df.loc[input_df[('Key',fold_column)].isin(fold),('Meta','Fold')] = i

    # Save output
    input_df.to_parquet(args.output)


if __name__ == '__main__':
    main()
