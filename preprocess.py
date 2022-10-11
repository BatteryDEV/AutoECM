import numpy as np
import pandas as pd
import json
import os
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import LabelEncoder
from utils_preprocessing import preprocess_data
from utils_preprocessing import unwrap_df
from utils_preprocessing import exclude_outlier
from utils_preprocessing import process_batch_element_params_str 
from utils_preprocessing import process_batch_element_params


def extract_tsfresh(df, df_test, output_path, le, if_exclude_outlier=False):
    """Extracts the tsfresh features from the raw data and saves them to a file.
    """

    if if_exclude_outlier:
        df, df_test = exclude_outlier(df, df_test)
        output_path = output_path.replace(".", "ou.")

    # Resulting dataframes do not have labels! They only contain the id, freq, z_real, z_imag columns!
    df_ts = unwrap_df(df)
    df_ts_test = unwrap_df(df_test)

    # Creating labels y
    df_x = pd.DataFrame(index=np.unique(df_ts["id"]))
    df_x_test = pd.DataFrame(index=np.unique(df_ts_test["id"]))

    df_y = pd.Series(data=(le.transform(df["Circuit"])), index=np.unique(df_ts["id"]))
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
    print("Saving tsfresh feature data...")
    np.savetxt(output_path, np.c_[np.array(X), np.array(df_y)], delimiter=',')
    np.savetxt(output_path.replace("train", "test"), np.c_[np.array(X_test), np.array(df_y_test)], delimiter=',')   
    # Save the feature names as json 
    with open(output_path.replace("train", "feature_names").replace('.csv', '.json'), 'w') as f:
        json.dump(list(X.columns), f)
        
    return

def extract_raw_interpolated(df, df_test, output_path, le, if_exclude_outlier=False):
    """Extracts the raw interpolated spectra from the raw data and saves them to a file."""

    if if_exclude_outlier:
        df, df_test = exclude_outlier(df, df_test)
        output_path = output_path.replace(".csv", "_ourem.csv")

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

def plot_eis_nyquist_individual(df: pd.DataFrame, path, name='train', ):
    """Plot the nyquist diagram for all eis data in the dataframe"""

    nb_spectra = len(df)
    # nb_spectra = 1 
    for i in range(nb_spectra):
        fig, ax  = plt.subplots(figsize = (2, 2), dpi=64, frameon=False)

        ax.set_axis_off()
        ax.plot(df['zreal'][i], -df['zimag'][i], linewidth=3, color='black')

        # Approach to encode the frequency as a colour
        # real=df['zreal'][i]
        # imag=df['zimag'][i]
        # real_minmaxed = (real-np.min(real))/(np.max(real)-np.min(real))
        # imag_minmaxed = (imag-np.min(imag))/(np.max(imag)-np.min(imag))
        # plt.scatter(real_minmaxed, -imag_minmaxed,c=freq, cmap="viridis",norm=colors.LogNorm(vmin=10, vmax=100000))
        plt.tight_layout()
        try:
            plt.savefig(f'{path}{name}/{df["Circuit"][i]}/{i}.png', dpi=64, bbox_inches='tight', pad_inches=0)
        except:
            os.makedirs(f'{path}{name}/{df["Circuit"][i]}')
            plt.savefig(f'{path}{name}/{df["Circuit"][i]}/{i}.png', dpi=64, bbox_inches='tight', pad_inches=0)
        plt.close()
        if np.mod(i, 100) == 0:
            print(f'Processed {i} spectra out of {nb_spectra}')
        #plt.savefig(f'./{s.Circuit}/fig{i}', dpi=64)
        #plt.show()
    return

def transform_data_to_images(df, df_test, le, output_path, plot_spectra=True):
    """Transforms the raw data to images and saves them to a file."""

    # Now let's make the images
    plot_eis_nyquist_individual(df, output_path, name='train')
    plot_eis_nyquist_individual(df_test, output_path, name='test')
    return 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    train_data_fname = 'data/train_data.csv'
    test_data_fname = 'data/test_data.csv'
    output_path_tsfresh = 'data/train_tsfresh.csv'
    output_path_raw = 'data/train_data_inter.csv'
    output_path_cnn = 'data/images/'
    le_path = 'data/le_name_mapping.json'

    df = preprocess_data(train_data_fname)
    df_test = preprocess_data(test_data_fname)

    # Load label encoder
    with open(le_path, 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping['classes'] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping['classes'])

    # Creating labels y
    with open(le_path, 'r') as f:
        mapping = json.load(f)

    le = LabelEncoder()
    mapping['classes'] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping['classes'])
    
    # Extract features, by uncommenting the function you want to use
    print("Preprocessing data...")
    extract_raw_interpolated(df, df_test, output_path_raw, le, if_exclude_outlier=True)
    
    # print("Preprocessing data tsfresh...")
    # extract_tsfresh(df, df_test,  output_path_tsfresh, if_exclude_outlier=False)
    # extract_tsfresh(df, df_test,  output_path_tsfresh, if_exclude_outlier=True)
    # print("Preprocessing data images...")
    # transform_data_to_images(df, df_test, le, output_path_cnn, plot_spectra=True)

    print("Done.")