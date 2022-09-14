import numpy as np
import pandas as pd
import json
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import LabelEncoder
from eis_preprocessing import preprocess_data
from eis_preprocessing import unwrap_df
from eis_preprocessing import exclude_outlier
from eis_preprocessing import plot_all_spectra

def extract_umap(traindata_filename_path, testdata_filename_path, output_path, if_exclude_outlier=False):
    """Extracts the umap features from the raw data and saves them to a file."""
    df = preprocess_data(traindata_filename_path)
    df_test = preprocess_data(testdata_filename_path)

    if if_exclude_outlier:
        df, df_test = exclude_outlier(df, df_test)
        output_path = output_path.replace(".csv", "ou.csv")

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
    return

def extract_raw_interpolated(traindata_filename_path, testdata_filename_path, output_path, if_exclude_outlier=False):
    """Extracts the raw interpolated spectra from the raw data and saves them to a file."""
    df = preprocess_data(traindata_filename_path)
    df_test = preprocess_data(testdata_filename_path)

    # Plot the raw data
    fig0, ax0 = plot_all_spectra(df, ax=None, plot_real=True, color="k")
    fig1, ax1 = plot_all_spectra(df, ax=None, plot_real=False, color="k")
    fig2, ax2 = plot_all_spectra(df_test, ax=None, plot_real=True, color="k")
    fig3, ax3 = plot_all_spectra(df_test, ax=None, plot_real=False, color="k")
    if if_exclude_outlier:
        df, df_test = exclude_outlier(df, df_test)
        output_path = output_path.replace(".csv", "ou.csv")

        plot_all_spectra(df, fig=fig0, ax=ax0, plot_real=True, color="blue")
        plot_all_spectra(df, fig=fig1, ax=ax1, plot_real=False, color="blue")
        plot_all_spectra(df_test, fig=fig2, ax=ax2, plot_real=True, color="blue")
        plot_all_spectra(df_test, fig=fig3, ax=ax3, plot_real=False, color="blue")
        
    # Creating labels y
    with open('XGB/data/le_name_mapping.json', 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()
    mapping['classes'] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping['classes'])

    df_y = pd.Series(data=(le.transform(df["Circuit"])), index=df.index)
    df_y_test = pd.Series(data=(le.transform(df_test["Circuit"])), index=df_test.index)

    # Exctract the spectra 
    X_train = np.zeros((len(df), 2*len(df.zimag[0])))
    X_test = np.zeros((len(df_test), 2*len(df_test.zimag[0])))
    for i in range(len(df)):
        X_train[i, :] = np.concatenate((df.zreal[i], -df.zimag[i]))
    for i in range(len(df_test)):
        X_test[i, :] = np.concatenate((df_test.zreal[i], -df_test.zimag[i]))
        
    data_train = np.concatenate((X_train, np.array(df_y).reshape(-1,1)), axis=1)
    data_test = np.concatenate((X_test, np.array(df_y_test).reshape(-1,1)), axis=1)


    # Save data
    print("Saving raw interpolated data...")
    np.savetxt(output_path, data_train, delimiter=',')
    np.savetxt(output_path.replace("train", "test"), data_test, delimiter=',')
    print("Done.")    
    return

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    train_data_fname = 'XGB/data/train_data_newspl.csv'
    test_data_fname = 'XGB/data/test_data_newspl.csv'
    output_path_umap = 'XGB/data/umap_train.csv'
    output_path_raw = 'XGB/data/train_data_newspl_inter.csv'
    extract_umap(train_data_fname, test_data_fname, output_path_umap, if_exclude_outlier=True)
    # extract_raw_interpolated(train_data_fname, test_data_fname, output_path_raw, if_exclude_outlier=True)
