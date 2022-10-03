import json
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
import datetime
import os

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

import xgboost as xgb

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters

from utils_preprocessing import preprocess_data
from utils_preprocessing import process_batch_element_params_str 
from utils_preprocessing import process_batch_element_params
from utils_preprocessing import eis_label_encoder
from preprocess import unwrap_df
 
from clf_tsfresh_xgb import load_features_le


def create_pipeline_reg(df_ts_):
    """Creates the pipeline for the regression model.
    
    Parameters
    ----------
    df_ts_ : pandas.DataFrame
        The dataframe containing the unwrapped data for the given ECM type.
    
    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
        The pipeline for the regression model.
    """
    mdl = MultiOutputRegressor(estimator=xgb.XGBRegressor())
    regr = TransformedTargetRegressor(
        regressor=mdl,
        transformer=QuantileTransformer(n_quantiles=10, random_state=42)
    )
    pipe = Pipeline(
        [
            (
                "augmenter",
                FeatureAugmenter(
                    column_id="id",
                    column_sort="freq",
                    default_fc_parameters=ComprehensiveFCParameters(),
                ),
            ),
            ("regressor", regr),
        ]
    )
    pipe.set_params(augmenter__timeseries_container=df_ts_)
    return pipe

def write_param_str_factory(param_strs):
    def write_param_str(param_values):
        return ", ".join([f"{key}: {value}" for key, value in zip(param_strs, param_values)])
    return write_param_str

def fit_ecm_parameter_model(df, df_ts, output_dir):
    """Fits the ECM parameter model for each ECM type.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the raw data.
    df_ts : pandas.DataFrame
        The dataframe containing the unwrapped data.
    output_dir : str    
        The directory to save the model to.

    Returns
    -------
    None
    """
    for ecm in df["Circuit"].unique():
        print(f"Fitting regression for {ecm}")

        ## Subselect circuits of given ECM
        df_ecm = df[df["Circuit"] == ecm]
        df_ts_ = df_ts[df_ts["id"].isin(df_ecm.index)]

        df_x = pd.DataFrame(index=np.unique(df_ts_["id"]))
        df_y = pd.DataFrame(
            df_ecm["param_values"].to_list(),
            columns=df_ecm["param_strs"].loc[df_ecm.index[0]]
        )

        print("Starting regr. feature extraction and training pipeline")
        pipe = create_pipeline_reg(df_ts_)
        ## Evaluate features and fit XGBoost Model
        pipe.fit(df_x, df_y)

        print("trainign completed, writing model and predictions to disc")
        dump(pipe, f"{output_dir}/xgb-ts-feat-reg-{ecm}-data-pipeline.joblib")

        with open(f"{output_dir}/{ecm}_param-names.txt", "w") as f:
            json.dump(df_ecm["param_strs"].loc[df_ecm.index[0]].tolist(), f)

    return 

def predict_ecm_parameters(df, df_ts, clf_model, X, label_dict, output_dir, train): 
    """Predicts the ECM parameters for the given data and model. 
    Saves the predictions to a file.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the raw data.
    df_ts : pandas.DataFrame
        The dataframe containing the unwrapped data.
    clf_model : xgboost.XGBClassifier
        The trained model.
    X : ndarray   
        The feature matrix.
    label_dict : dict
        The dictionary containing the label mapping.
    output_dir : str
        The directory to save the predictions to.
    train : bool
        Whether the data is from the training set.
    
    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the raw data and the predictions.
    """

    print("Predict class")
    df["pred_label"] = clf_model.predict(X)
    df["pred_circuit"] = df["pred_label"].apply(int).apply(str).map(label_dict)

    df["Parameters"] = [""] * len(df)

    for ecm in label_dict.values():
        print(f"Predict model params for {ecm}")
        reg_pipe = load(f"{output_dir}/xgb-ts-feat-reg-{ecm}-data-pipeline.joblib")

        with open(f"{output_dir}/{ecm}_param-names.txt", "r") as f:
            param_strs = json.load(f)

        df_ecm = df[df["pred_circuit"] == ecm]

        if len(df_ecm) == 0:
            continue

        df_ts_ = df_ts[df_ts["id"].isin(df_ecm.index)]

        df_x = pd.DataFrame(index=np.unique(df_ts_["id"]))

        reg_pipe.set_params(augmenter__timeseries_container=df_ts)
        pred_values = reg_pipe.predict(df_x)

        f = write_param_str_factory(param_strs)

        df.loc[df_ecm.index, "pred_parameters"] = [f(params) for params in pred_values]

    print('Save predictions')
    if train:
        df[["pred_circuit", "pred_parameters"]].to_csv(f"{output_dir}/train_data_predictions.csv")
    else:
        df[["pred_circuit", "pred_parameters"]].to_csv(f"{output_dir}/test_data_predictions.csv")

    return df

def dummy_model(): 
    # dummy model, no quantile transform
    ppls_0 = list()
    for df_ in dfs:
        print(" ")
        print("Fitting regression for %s" % df_["Circuit"].loc[df_.index[0]])
        df_ts_ = df_ts[df_ts["id"].isin(df_.index)]
        df_x = pd.DataFrame(index=np.unique(df_ts_["id"]))
        df_y = pd.DataFrame(df_['param_values'].to_list(), columns=df_["param_strs"].loc[df_.index[0]])
        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
        # transformer mask, tell the transformer which parameters to skip over
        mask = [n for n, x in enumerate(df_["param_strs"].loc[df_.index[0]]) if '_t' in x]
        mdl = DummyRegressor(strategy='median')
        if quantile: 
              regr = TransformedTargetRegressor(regressor=mdl,
                                    transformer=QuantileTransformer(n_quantiles=10, random_state=0))
            #regr = TransformedTargetRegressor(regressor=mdl, 
            #                                  func=partial(transform_params, mask_skip=mask),
            #                                  inverse_func=partial(invert_params, mask_skip=mask))
        else:
            regr = mdl
        ppl = Pipeline([
                ('regressor', regr)
            ])

        ppl.fit(X_train, y_train)
        y_pred = ppl.predict(X_test)
        print("MAE:%.4g" % mean_absolute_error(y_test, y_pred))
        p_strs = df_["param_strs"].loc[df_.index[0]]
        for i in np.arange(len(p_strs)):
            print("%s MAE:%.4g" % (p_strs[i], mean_absolute_error(y_test[p_strs[i]], y_pred[:,i])))
        ppls_0.append(ppl)
    return None

def MAE_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


if __name__ == "__main__":

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
    
    fit_ecm_parameter_model(df, df_ts, output_dir)

    print("Load clf model")
    le, label_dict = eis_label_encoder(le_f)
    del(label_dict['classes'])  # Delete the classes key

    clf_model = load(clf_model_fname)

    # predict_ecm_parameters(df, df_ts, clf_model, Xtsf_train, label_dict, output_dir, train=True)
    predict_ecm_parameters(df_test, df_ts_test, clf_model, Xtsf_test, label_dict, output_dir, train=False)

    mae
    print("Done")