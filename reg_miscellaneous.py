import pandas as pd
from sklearn.pipeline import Pipeline
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters
from utils_preprocessing import eis_label_encoder

def fit_ecm_parameter_model(df, df_ts, quantile_transformer, use_dummy_model, output_dir):
    """Fits the ECM parameter model for each ECM type.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the raw data.
    df_ts : pandas.DataFrame
        The dataframe containing the unwrapped data.
    quantile_transformer : bool
        Whether to use a quantile transformer of the target values or not
        prior to fitting the model.
    use_dummy_model : bool
        Whether to use a dummy model. If true dummy model is used
        instead of the XGBoost model.
    output_dir : str    
        The directory to save the model to.

    Returns
    -------
    None
    """
    if use_dummy_model:
        model_name = "dummy"
    else:
        model_name = "xgb-ts-feat"
    
    if quantile_transformer:
        model_name += "-qt"

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
        pipe = create_pipeline_reg(df_ts_, quantile_transformer, use_dummy_model)
        ## Evaluate features and fit XGBoost Model
        pipe.fit(df_x, df_y)

        print("trainign completed, writing model and predictions to disc")
        ## Save model

        dump(pipe, f"{output_dir}/{model_name}-reg-{ecm}-data-pipeline.joblib")
        with open(f"{output_dir}/reg-{ecm}_param-names.txt", "w") as f:
            json.dump(df_ecm["param_strs"].loc[df_ecm.index[0]].tolist(), f)

    return 

def predict_ecm_parameters(df, df_ts, X, label_dict, output_dir, predict_train_set=False):
    """Predicts the ECM parameters for the given data and model. 
    Saves the predictions to a file.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the raw data.
    df_ts : pandas.DataFrame
        The dataframe containing the unwrapped data.
    X : ndarray   
        The feature matrix.
    label_dict : dict
        The dictionary containing the label mapping.
    output_dir : str
        The directory to save the predictions to.
    predict_train_set : bool
        Whether the data is from the training set.
    
    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the raw data and the predictions.
    """

    # TO DO: Restructure this function for new stragegy
    df["pred_parameters"] = [""] * len(df)

    for ecm in label_dict.values():
        print(f"Predict model params for {ecm}")
        reg_pipe = load(f"{output_dir}/xgb-ts-feat-reg-{ecm}-data-pipeline.joblib")

        with open(f"{output_dir}/{ecm}_param-names.txt", "r") as f:
            param_strs = json.load(f)

        df_ecm = df[df["Circuit"] == ecm]

        df_ts_ = df_ts[df_ts["id"].isin(df_ecm.index)]

        df_x = pd.DataFrame(index=np.unique(df_ts_["id"]))

        reg_pipe.set_params(augmenter__timeseries_container=df_ts)
        pred_values = reg_pipe.predict(df_x)

        f = write_param_str_factory(param_strs)

        df.loc[df_ecm.index, "pred_parameters"] = [f(params) for params in pred_values]

    print('Save predictions')
    if predict_train_set:
        df[["pred_circuit", "pred_parameters"]].to_csv(f"{output_dir}/train_data_predictions.csv")
    else:
        df[["pred_circuit", "pred_parameters"]].to_csv(f"{output_dir}/test_data_predictions.csv")

    return df

def create_pipeline_reg(df_ts_, quantile_transformer=True, use_dummy_model=False):
    """Creates the pipeline for the regression model.
    
    Parameters
    ----------
    df_ts_ : pandas.DataFrame
        The dataframe containing the unwrapped data for the given ECM type.
    quantile_transformer : bool
        Whether to use a quantile transformer of the target values or not
        prior to fitting the model.
    use_dummy_model : bool
        Whether to use a dummy model. If true dummy model is used
        instead of the XGBoost model.

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
        The pipeline for the regression model.
    """
    if not use_dummy_model:
        mdl = MultiOutputRegressor(estimator=xgb.XGBRegressor())
    else:
        mdl = DummyRegressor(strategy='median')

    if quantile_transformer:
        regr = TransformedTargetRegressor(
            regressor=mdl,
            transformer=QuantileTransformer(n_quantiles=10, random_state=42)
            )
    else:
        regr = mdl

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


if __name__ == "__main__":
       # Full XGB model
    print("Fitting full XGB model")
    fit_ecm_parameter_model(df, df_ts, 
        quantile_transformer=False, use_dummy_model=False,
        output_dir=output_dir)
    # Quantile XGB Model 
    print("Fitting quantile XGB model")
    fit_ecm_parameter_model(df, df_ts, 
        quantile_transformer=True, use_dummy_model=False,
        output_dir=output_dir)
    # Dummy Model
    print("Fitting dummy model")
    fit_ecm_parameter_model(df, df_ts, 
        quantile_transformer=False, use_dummy_model=True,
        output_dir=output_dir)
    # Quntile Dummy Model
    print("Fitting quantile dummy model")
    fit_ecm_parameter_model(df, df_ts, 
        quantile_transformer=True, use_dummy_model=True,
        output_dir=output_dir)

    # Predictions
    # predict_ecm_parameters(df, df_ts, clf_model, Xtsf_train, label_dict, output_dir, train=True)
    # predict_ecm_parameters(df_test, df_ts_test, clf_model, Xtsf_test, label_dict, output_dir, train=False)


    
    # np.savetxt(f"{output_dir}/{ecm}_train_data_predictions.csv", pred_parameters_train, delimiter=",")
    # np.savetxt(f"{output_dir}/{ecm}_train_data_qsparams.csv", y_train, delimiter=",")
    # np.savetxt(f"{output_dir}/{ecm}_test_data_predictions.csv", pred_parameters_test, delimiter=",")
    # np.savetxt(f"{output_dir}/{ecm}_test_data_qsparams.csv", y_test, delimiter=",")