from argparse import ArgumentParser
from os import makedirs, path
import pandas as pd
from tqdm import tqdm
from gilg_utils.general import load_yaml
from gilg_utils.models import PytorchNeuralNetworkRegressor
import polars as pl
pd.set_option('future.no_silent_downcasting', False)

def add_multicolumn(input_df,model,label_column):
    for f in model.features:
        if f not in input_df.columns:
            input_df[f] = 0
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

    input_df = input_df.fillna(0)
    input_df.columns = pd.MultiIndex.from_arrays([column_categories,input_df.columns],names=('Category','Column'))
    return input_df
    
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
    # input_df = input_df[input_df[('Key','date_id')].isin(sorted(list(set(input_df[('Key','date_id')])))[:4])]
    makedirs(path.dirname(args.output), exist_ok=True)
    input_df.columns = input_df.columns.droplevel('Category')
    label_column = 'responder_6'

    # Load Model
    model = PytorchNeuralNetworkRegressor()
    model_path = path.join(args.model,f'PytorchNeuralNetworkRegressor')
    model.load(model_path)

    # Define predict function
    global lags_
    lags_ = None
    global time_lags_
    time_lags_ = None
    global prediction_lags_
    prediction_lags_ = None
    def predict(test: pd.DataFrame, lags: pd.DataFrame | None) -> pd.DataFrame | pd.DataFrame:
        """Make a prediction."""
        # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
        # Use them as extra features, if you like.
        global lags_
        global time_lags_
        global prediction_lags_
        # print('New Predict')
        # print('global pred lags')
        # print(prediction_lags_)
        if lags is not None:
            lags_ = lags
        lags = lags_
            
        # Compute difs
        input_df = test.to_pandas()
        label_column = 'responder_6'
        time_lag_df = pd.DataFrame(input_df['symbol_id'])
        if time_lags_ is None:
            pass
        else:
            time_lag_df = time_lag_df.merge(time_lags_.to_pandas(),how='left',on='symbol_id').fillna(0)
        prediction_lag_df = pd.DataFrame(input_df['symbol_id'])
        if prediction_lags_ is None:
            prediction_lag_df[label_column] = 0
        else:
            prediction_lag_df = prediction_lag_df.merge(prediction_lags_.to_pandas(),how='left',on='symbol_id').fillna(0)

        input_df = input_df.fillna(0)
        input_df = add_multicolumn(input_df,model,label_column)
        time_lag_df = add_multicolumn(time_lag_df,model,label_column)
        input_df['Data'] = input_df['Data'].values - time_lag_df['Data'].values

        # Make predictions
        pred = model.predict(input_df)
        predictions = pd.DataFrame()
        predictions['row_id'] = input_df[('Key','row_id')].values
        if time_lags_ is None:
            predictions[label_column] = 0
        else:
            predictions[label_column] = prediction_lag_df[label_column] + pred.values
        
        if isinstance(predictions, pd.DataFrame):
            assert (predictions.columns == ['row_id', 'responder_6']).all()
        else:
            raise TypeError('The predict function must return a DataFrame')
        # Confirm has as many rows as the test data.
        assert len(predictions) == len(test)
        predictions = pl.from_pandas(predictions)

        predictions_lag = pd.DataFrame()
        predictions_lag['symbol_id'] = input_df[('Key','symbol_id')].values
        if time_lags_ is None:
            predictions_lag[label_column] = 0
        else:
            predictions_lag[label_column] = prediction_lag_df[label_column] + pred.values
        predictions_lag = pl.from_pandas(predictions_lag)
        prediction_lags_ = predictions_lag
        time_lags_ = test
        # print('model diff pred')
        # print(pred)
        # print('return')
        # print(predictions)
        # print('global pred lags')
        # print(prediction_lags_)
        # print(test)
        # print(predictions_lag)
        # print(predictions)
        # exit()
        return predictions
        
    # Test model
    total_pred = []
    past_date_df = None
    for date_id in tqdm(sorted(list(set(input_df['date_id'])))):
        date_df = input_df[input_df['date_id']==date_id].copy()
        for time_id in sorted(list(set(date_df['time_id']))):
            if time_id==0:
                if past_date_df is None:
                    lags = None
                else:
                    lags = past_date_df.copy()
                    new_columns = []
                    for c in lags.columns:
                        if '_id' in c:
                            new_columns.append(c)
                        else:
                            new_columns.append(c+'_lag_1')
                    lags.columns = new_columns
                    lags = pl.from_pandas(lags)
                
            else:
                lags = None
            time_df = date_df[date_df['time_id']==time_id].copy()
            time_df['row_id'] = time_df.index.values
            time_df = pl.from_pandas(time_df)
            time_pred = predict(time_df,lags)
            total_pred.append(time_pred.to_pandas())
        past_date_df = date_df.copy()
    total_pred = pd.concat(total_pred,axis=0)
    total_pred = total_pred.set_index('row_id')
    total_pred.columns = pd.MultiIndex.from_arrays([['Predictions'],['Prediction']],names=('Category','Column'))

    # Add column levels back
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
    input_df = pd.concat([input_df,total_pred],axis=1)
    input_df = input_df[['Key','Predictions','Meta','Label']]
    input_df.to_parquet(args.output)

if __name__ == '__main__':
    main()