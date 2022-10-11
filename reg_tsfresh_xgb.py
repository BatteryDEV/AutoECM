import json
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
import datetime
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

from utils_preprocessing import preprocess_data
from utils_preprocessing import process_batch_element_params_str 
from utils_preprocessing import process_batch_element_params
from preprocess import unwrap_df
 
from clf_tsfresh_xgb import load_features_le


def write_param_str_factory(param_strs):
    def write_param_str(param_values):
        return ", ".join([f"{key}: {value}" for key, value in zip(param_strs, param_values)])
    return write_param_str

### Try again with saved features
def fit_ecm_parameter_model(df, df_test, quantile_transformer, use_dummy_model, output_dir):
    """Fits the ECM parameter model for each ECM type.
    loads features directly from file"""
        
    # Select regressor
    if not use_dummy_model:
        mdl = MultiOutputRegressor(estimator=xgb.XGBRegressor())
        model_name = "xgb-ts-feat"
    else:
        mdl = DummyRegressor(strategy='median')
        model_name = "dummy"

    if quantile_transformer:
        regr = TransformedTargetRegressor(
            regressor=mdl,
            transformer=QuantileTransformer(n_quantiles=10, random_state=0)
            )
        model_name += "-qt"
    else:
        regr = mdl
    
    le_path = 'data/le_name_mapping.json'
    # Creating labels y
    with open(le_path, 'r') as f:
        mapping = json.load(f)
    mapping['classes'] = [mapping[str(int(i))] for i in range(9)]
    df["param_values_pred"] = [""] * len(df)
    df_test["param_values_pred"] = [""] * len(df_test)

    for ecm in mapping['classes']:
        print(f"ECM: {ecm}, Model: {model_name}")
        # Load features (due to nans in the data loadtxt or genfromtxt might fail
        try:
            train_data = np.loadtxt(f"data/reg-tsfresh/{ecm}_data_train.csv")
        except:
            train_data = np.genfromtxt(f"data/reg-tsfresh/{ecm}_data_train.csv", delimiter=",")
        try:
            test_data = np.loadtxt(f"data/reg-tsfresh/{ecm}_data_test.csv")
        except:
            test_data = np.genfromtxt(f"data/reg-tsfresh/{ecm}_data_test.csv", delimiter=",")

        # Load param names from json file
        with open(f"data/reg-tsfresh/{ecm}_param_names.json", 'r') as f:
            param_strs = json.load(f)

        nb_params = len(param_strs)
        # Preprocess data
        X_train = train_data[:,0:-nb_params]
        y_train = train_data[:,-nb_params:]
        X_test = test_data[:,0:-nb_params]
        y_test = test_data[:,-nb_params:]

        # Train model
        regr.fit(X_train, y_train)
        # Save model
        dump(regr, f"{output_dir}/models/{model_name}-reg-{ecm}-reg-model.joblib")

        # Make predictions on train set 
        pred_values_train = regr.predict(X_train)
        # Make predictions on test set
        pred_values_test = regr.predict(X_test)

        df_ecm = df[df["Circuit"] == ecm]
        df.loc[df_ecm.index, "param_values_pred"] = [[pred_values_train[i, :]] for i in range(pred_values_train.shape[0])]
        df_test_ecm = df_test[df_test["Circuit"] == ecm]
        df_test.loc[df_test_ecm.index, "param_values_pred"] = [[pred_values_test[i, :]] for i in range(pred_values_test.shape[0])]

        # print('Save predictions')
        # np.savetxt(pred_values_train, f"{output_dir}/predictions/{model_name}_{ecm}_train_data_predictions.csv")
        # np.savetxt(pred_values_test, f"{output_dir}/predictions/{model_name}_{ecm}_test_data_predictions.csv")

    path_train = f"{output_dir}/predictions/{model_name}_{ecm}_train_data_predictions.pkl"
    path_test = f"{output_dir}/predictions/{model_name}_test_data_predictions.pkl"
    df.to_pickle(path_train)
    df_test.to_pickle(path_test)

    score_qs_vs_pred(path_train, path_test, model_name)
    return

def score_qs_vs_pred(path_train, path_test, model_name):
    # f = write_param_str_factory(param_strs)
    # pred_parameters_train = [f(params) for params in pred_values_train]
    # pred_parameters_test = [f(params) for params in pred_values_test]
    df = pd.read_pickle(path_train)
    df_test = pd.read_pickle(path_test)
    df_results = pd.DataFrame()
    results = []
    for ecm in np.unique(df['Circuit']):
        df_ecm = df[df['Circuit'] == ecm]
        df_test_ecm = df_test[df_test['Circuit'] == ecm]

        y_true_train = np.stack(df_ecm['param_values'].values, axis=0).squeeze()
        y_pred_train = np.stack(df_ecm['param_values_pred'].values, axis=0).squeeze()
        y_true_test = np.stack(df_test_ecm['param_values'].values, axis=0).squeeze()
        y_pred_test = np.stack(df_test_ecm['param_values_pred'].values, axis=0).squeeze()

        mae_train = MAE_loss(y_true_train, y_pred_train)
        mae_test = MAE_loss(y_true_test, y_pred_test)

        print(f"ECM: {ecm}, MAE train: {mae_train}, MAE test: {mae_test}")
        for i, element in enumerate(df_ecm['param_strs'].iloc[0]):
            # mae_train_element = MAE_loss(y_true_train.apply(lambda x: x[i]), y_pred_train.apply(lambda x: x[i]))
            mae_test_element = MAE_loss(y_true_test[:, i], y_pred_test[:, i])
            res_element = [ecm, element, mae_test_element]
            results.append(res_element)
    # Save results
    df_results = pd.DataFrame(results, columns=['ecm', 'param', 'mae'])
    df_results.to_csv(f"{output_dir}/predictions/{model_name}_qs_vs_pres_mae_results.csv", index=False)
    return results

def score_eis_custom_cv_loss(path_train, path_test): 
    pass
    return

def MAE_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


if __name__ == "__main__":
    train_pred = True

    if train_pred:
        le_f = "data/le_name_mapping.json"
        train_data_fname = 'data/train_data.csv'
        test_data_fname = 'data/test_data.csv'
        clf_model_fname = 'results/clf/xgb/2022-09-30_11-59-51/model.joblib'

        # Load tsfresh data directly from file
        train_data_f = "data/train_tsfresh.csv"
        test_data_f = "data/test_tsfresh.csv"  
        Xtsf_train, ytsf_train, Xtsf_test, ytsf_test, le = load_features_le(train_data_f, test_data_f, le_f)

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"results/reg/xgb/{now_str}"
        os.mkdir(output_dir)
        os.mkdir(output_dir+"/models")
        os.mkdir(output_dir+"/predictions")

        df = preprocess_data(train_data_fname)
        df_test = preprocess_data(test_data_fname)

        df_ts = unwrap_df(df)
        df_ts_test = unwrap_df(df_test)

        df["param_strs"] = df.apply(
            lambda x: process_batch_element_params_str(x.Parameters), axis=1
        )
        df["param_values"] = df.apply(
            lambda x: process_batch_element_params(x.Parameters), axis=1
        )

        df_test["param_strs"] = df_test.apply(
            lambda x: process_batch_element_params_str(x.Parameters), axis=1
        )
        df_test["param_values"] = df_test.apply(
            lambda x: process_batch_element_params(x.Parameters), axis=1
        )
        # Fit and predict and save results
        fit_ecm_parameter_model(
            df, df_test, 
            quantile_transformer=False, use_dummy_model=False, 
            output_dir=output_dir)
        
        fit_ecm_parameter_model(
            df, df_test, 
            quantile_transformer=True, use_dummy_model=False, 
            output_dir=output_dir)
        
        fit_ecm_parameter_model(
            df, df_test, 
            quantile_transformer=False, use_dummy_model=True, 
            output_dir=output_dir)


    #if score:
    #    # Score predictions
    #    path_train = 'results/reg/xgb/2022-10-04_20-03-55/predictions/xgb-ts-feat_train_data_predictions.pkl'
    #    path_test = 'results/reg/xgb/2022-10-04_20-03-55/predictions/xgb-ts-feat_test_data_predictions.pkl'
    #    results = score_qs_vs_pred(path_train, path_test)

    # Score predictions with local eis package custom scoring function

    print("Done")