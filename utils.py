import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import shap
import umap


def plot_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    save: bool = False,
    title: Optional[str] = None,
    figname: str = "test_confusion",
) -> None:
    """Plot confusion matrix.

    Parameters
    ----------
    y_true: array-like
        True labels
    y_pred: array-like
        Predicted labels
    le: LabelEncoder
        Label encoder
    save: bool
        Save figure
    save_path: str
        Path to save figure
    figname: str
        Name of figure

    Returns
    -------
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=le.inverse_transform(list(range(9))),
        columns=le.inverse_transform(list(range(9))),
    )

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True)
    if title:
        plt.title(title)
    else:
        plt.title("Confusion Matrix")
    plt.ylabel("Given ECM")
    plt.xlabel("Predicted ECM")
    plt.tight_layout()
    if save:
        plt.savefig(figname + ".eps", dpi=300)
        plt.savefig(figname + ".jpg", dpi=300)
    plt.show()
    return


def calcualte_classification_report(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    le: LabelEncoder,
    save: bool = False,
    output_dir: str = "",
) -> None:
    """Calculate classification report for train and test data.

    Parameters
    ----------
    y_train: array-like
        True labels for train data
    y_train_pred: array-like
        Predicted labels for train data
    y_test: array-like
        True labels for test data
    y_test_pred: array-like
        Predicted labels for test data
    le: LabelEncoder
        Label encoder
    save: bool
        Save classification report and predictions
    output_dir: str
        Path to save classification report and predictions

    Returns
    -------
    None
    """
    acc_train = f1_score(y_train, y_train_pred, average="macro")
    acc_test = f1_score(y_test, y_test_pred, average="macro")

    print(f"F1 score train: {acc_train:.3f}")
    print(f"F1 score test: {acc_test:.3f}")

    # Make classifcation report
    cl_report = classification_report(
        y_test, y_test_pred, target_names=le.classes_, digits=4
    )
    print(cl_report)

    if save:
        # Save classfication report
        with open(f"{output_dir}/report.txt", "w") as f:
            f.write(cl_report)
        # Save predcitions
        np.savetxt(f"{output_dir}/pred_test.txt", y_test_pred, fmt="%d")

    return


def shap_feature_analysis(
    mdl,
    x_transformed: np.ndarray,
    le: LabelEncoder,
    save: bool = False,
    output_dir: str = "",
    max_display: int = 12,
) -> None:
    """Use SHAP to investigate feature importance and dependence on critical features for
    making predictions on the test set.

    Parameters
    ----------
    ppl: Pipeline
        Pipeline with fitted model
    X: array-like
        Test data
    le: LabelEncoder
        Label encoder
    save: bool
        Save figure

    Returns
    -------
    None
    """
    # SHAP objects
    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(x_transformed)

    shap.summary_plot(
        shap_values,
        x_transformed,
        plot_size=(20, 5),
        max_display=max_display,
        class_names=le.classes_,
        class_inds="original",
        show=True,
    )

    if save:
        plt.savefig(f"{output_dir}/shap_summary_bar.eps", dpi=300)
        plt.savefig(f"{output_dir}/shap_summary_bar.jpg", dpi=300)

    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(x_transformed.columns, sum(vals))),
        columns=["col_name", "feature_importance_vals"],
    )
    feature_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    feature_importance.iloc[0:5]

    # only use 5 most important average features
    imp_feat_ind = feature_importance.index[0:5].values
    imp_feat_cols = feature_importance.col_name[0:5]
    for i, label in enumerate(le.classes_):
        plt.subplot(3, 3, i + 1)
        shap.summary_plot(
            shap_values[i][:, imp_feat_ind],
            x_transformed[imp_feat_cols],
            show=False,
            feature_names=1 + np.arange(5),
            color_bar_label=None,
        )
        plt.xlabel(None)
        plt.title(label)

    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/shap_class_specific.eps", dpi=300)
        plt.savefig(f"{output_dir}/shap_class_specific.jpg", dpi=300)
    plt.show()
    return


def plot_freq_range(df: pd.DataFrame, save: bool = False, verbose: int = 1) -> None:
    """Show the frequency ranges for each circuit type (frequency range data leakage).

    Parameters
    ----------
    df: DataFrame
        Dataframe with EIS data
    save: bool
        Save figure
    verbose: bool
        Print frequency information
    """

    circuits, indices = np.unique(df.Circuit, return_index=True)
    df_unique = df.iloc[indices]

    for i in df_unique.index:
        plt.vlines(
            df_unique.Circuit[i],
            np.min(df_unique.freq[i]),
            np.max(df_unique.freq[i]),
        )

    plt.yscale("log")
    plt.ylabel("Frequency range (Hz)")
    plt.xticks(rotation="vertical")
    plt.xlabel("Equivalent circuit model")

    plt.tight_layout()
    if save:
        plt.savefig("figures/frequency_ranges.eps", dpi=300)
        plt.savefig("figures/frequency_ranges.jpg", dpi=300)
    plt.show()
    if verbose:
        print(circuits, indices)
        f_max = df.freq.apply(np.max).min()
        f_min = df.freq.apply(np.min).max()
        print(f"Maximal minimum freq: {f_min}, minimal maximum freq: {f_max}")
    return


def umap_plots(df_sorted: pd.DataFrame, save: int = 0, random_state: int = 42) -> None:
    """Make a UMAP plot of the data set.

    Parameters
    ----------
    df_sorted: DataFrame
        Dataframe with sorted EIS data
    save: bool
        Save figure

    Returns
    -------
    None
    """
    # Just get the impedance values into a numpy array
    d_z = np.zeros((df_sorted.shape[0], 60))
    for i in df_sorted.index:
        d = df_sorted.iloc[i]
        d_z[i, :] = np.concatenate((d["zreal"], d["zimag"]))

    # UMAP transform
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, random_state=random_state)
    embeds = reducer.fit_transform(d_z)

    ax = sns.scatterplot(
        x=embeds[:, 0], y=embeds[:, 1], hue=df_sorted.Circuit, palette="cividis"
    )
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()
    if save:
        plt.savefig("figures/umap.eps", dpi=300)
        plt.savefig("figures/umap.jpg", dpi=300)
    plt.show()

    x = np.linspace(0.0, 1.0, 9)
    colors = cm.get_cmap("tab10")(x)[np.newaxis, :, :3]

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))
    ax = ax.ravel()

    circuits, indices = np.unique(df_sorted.Circuit, return_index=True)
    for i, circuit in enumerate(circuits):
        mask = df_sorted.Circuit == circuit
        ax[i].hexbin(embeds[mask, 0], embeds[mask, 1], gridsize=25, cmap="magma")
        # ax[i].hexbin(embeds[mask,0], embeds[mask,1], color=colors[0,i,:])
        # ax[i].plot(embeds[mask,0], embeds[mask,1], '.', markersize=4, color=colors[0,i,:])
        ax[i].set_xlabel("Component 1")
        ax[i].set_ylabel("Component 2")
        ax[i].set_title(circuit)

    plt.tight_layout()
    if save:
        plt.savefig("figures/umap_separated.eps", dpi=300)
        plt.savefig("figures/umap_separated.jpg", dpi=300)
    plt.show()
    return
