import json
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
import argparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

import xgboost as xgb

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters

from utils_preprocessing import preprocess_data
#from eis_preprocessing import process_batch_element_params_str 
#from eis_preprocessing import process_batch_element_params
#from eis_preprocessing import unwrap_df


def create_pipeline_clf(df_ts):
    ## Define sklearn pipeline
    pipe = Pipeline(
        [
            (
                "augmenter",
                RelevantFeatureAugmenter(
                    column_id="id",
                     column_sort="freq",
                    default_fc_parameters=ComprehensiveFCParameters(),
                ),
            ),
            ("classifier", xgb.XGBClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    pipe.set_params(augmenter__timeseries_container=df_ts);
    return pipe

def create_pipeline_reg(df_ts):
    ## Define sklearn pipeline
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIS spectra processing with ML")
    parser.add_argument(
        "--mode", 
        help="Only 'train', 'pred' or 'all' are valid inputs. You need to train the model beofre calling prediction\
            if you pass no argument, the cleassifier and the predictor will be trained and predicitons will be made")
    parser.add_argument(
        "--model_type",
        help="3 options implemented: 'rf' for random forest, 'xgb' for xgboost, 'cnn' for convolutional neural network")
    
    args = parser.parse_args()

    if args.mode:
        mode = args.mode
    else: 
        mode = "train_clf"

    if args.model_type:
        m_path = args.modelpath
     
    m_path = "models/xgb/"    
    d_path = "data/"
    
    pred_dfile = d_path + args.pred_data 
    pred_dfile =d_path + 'test_data.csv'

    print("Loading data...")
    
    
    if  mode=="train_clf" or  mode=="train_reg":
        df = preprocess_data(d_path + "train_data.csv")
    elif mode=="pred":
        df = preprocess_data(pred_dfile)



    if mode=="train_clf":
        ## Unwrap data frame for tsfresh
        df_ts = unwrap_df(df)
        df_x = pd.DataFrame(index=np.unique(df_ts["id"]))

        ## Encode circuit labels as strings
        le = LabelEncoder()
        df_y = pd.Series(data=(le.fit_transform(df["Circuit"])), index=np.unique(df_ts["id"]))

        print("Starting class. feature extraction and training pipeline")
        pipe = create_pipeline_clf(df_ts)
        ## Evaluate features and fit XGBoost Model
        pipe.fit(df_x, df_y)

        print("trainign completed, writing model and predictions to disc")
        dump(pipe, f"{m_path}clf/xgb-ts-feat-clf-all-data-pipeline.joblib")

        label_dict = dict(zip(list(range(9)), le.inverse_transform(list(range(9)))))

        with open(f"{m_path}labels.json", "w") as outfile:
            json.dump(label_dict, outfile)

    if mode=="train_reg": 
        df["param_strs"] = df.apply(
            lambda x: process_batch_element_params_str(x.Parameters), axis=1
        )
        df["param_values"] = df.apply(
            lambda x: process_batch_element_params(x.Parameters), axis=1
        )

        ## Unwrap data frame for tsfresh
        df_ts = unwrap_df(df)

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
            pipe = create_pipeline_reg(df_ts)
            ## Evaluate features and fit XGBoost Model
            pipe.fit(df_x, df_y)

            print("trainign completed, writing model and predictions to disc")
            dump(pipe, f"{m_path}reg/xgb-ts-feat-reg-{ecm}-data-pipeline.joblib")

            with open(f"{m_path}{ecm}_param-names.txt", "w") as f:
                json.dump(df_ecm["param_strs"].loc[df_ecm.index[0]].tolist(), f)
    
    if mode=="pred":
        ## Unwrap data frame for tsfresh
        df_ts = unwrap_df(df)
        df_x = pd.DataFrame(index=np.unique(df_ts["id"]))

        print("Load clf model")
        clf_pipe = load(f"{m_path}clf/xgb-ts-feat-clf-all-data-pipeline.joblib")
        
        clf_pipe.set_params(augmenter__timeseries_container=df_ts)
        with open(f"{m_path}labels.json", "r") as f:
            label_dict = json.load(f)

        print("Predict class")
        df["pred_label"] = clf_pipe.predict(df_x)
        df["Circuit"] = df["pred_label"].apply(str).map(label_dict)

        df["Parameters"] = [""] * len(df)

        for ecm in label_dict.values():
            print(f"Predict model params for {ecm}")
            reg_pipe = load(f"{m_path}reg/xgb-ts-feat-reg-{ecm}-data-pipeline.joblib")

            with open(f"{m_path}{ecm}_param-names.txt", "r") as f:
                param_strs = json.load(f)

            df_ecm = df[df["Circuit"] == ecm]

            if len(df_ecm) == 0:
                continue

            df_ts_ = df_ts[df_ts["id"].isin(df_ecm.index)]

            df_x = pd.DataFrame(index=np.unique(df_ts_["id"]))

            reg_pipe.set_params(augmenter__timeseries_container=df_ts)
            pred_values = reg_pipe.predict(df_x)

            f = write_param_str_factory(param_strs)

            df.loc[df_ecm.index, "Parameters"] = [f(params) for params in pred_values]
        print('Save predicrtions')
        df[["Circuit", "Parameters"]].to_csv(m_path+"submission.csv")
