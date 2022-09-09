import numpy as np
import pandas as pd
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import LabelEncoder
from eis_preprocessing import preprocess_data
from eis_preprocessing import unwrap_df

def main(traindata_filename_path, testdata_filename_path, output_path):
    df = preprocess_data(traindata_filename_path)
    df_test = preprocess_data(testdata_filename_path)

    # Resulting dataframes do not have labels! They only contain the id, freq, z_real, z_imag columns!
    df_ts = unwrap_df(df)
    df_ts_test = unwrap_df(df_test)

    # Creating labels y
    df_x = pd.DataFrame(index=np.unique(df_ts["id"]))
    df_x_test = pd.DataFrame(index=np.unique(df_ts_test["id"]))

    le = LabelEncoder()
    df_y = pd.Series(data=(le.fit_transform(df["Circuit"])), index=np.unique(df_ts["id"]))
    df_y_test = pd.Series(data=(le.transform(df_test["Circuit"])), index=np.unique(df_ts_test["id"]))

    augmenter = RelevantFeatureAugmenter(
                    column_id="id",
                    column_sort="freq",
                    default_fc_parameters=ComprehensiveFCParameters(),
                )
    augmenter.set_params(timeseries_container=df_ts)
    X = augmenter.fit_transform(df_x, df_y)
    augmenter.set_params(timeseries_container=df_ts_test)
    X_test = augmenter.transform(df_x_test)

    # Save data
    print("Saving umap feature data...")
    np.savetxt(output_path, np.c_[np.array(X), np.array(df_y)], delimiter=',')
    np.savetxt(output_path.replace("train", "test"), np.c_[np.array(X_test), np.array(df_y_test)], delimiter=',')   
    print("Done.")    
    pass

if __name__ == "__main__":
    train_data_fname = 'XGB/data/train_data.csv'
    test_data_fname = 'XGB/data/test_data_hold_out_labeled.csv'
    main(train_data_fname, test_data_fname, 'XGB/data/umap_train.csv')