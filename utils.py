import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional
from sklearn.preprocessing import LabelEncoder
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
        plt.savefig(f"{output_dir}/shap_summary_bar.eps")
        plt.savefig(f"{output_dir}/shap_summary_bar.pdf")
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


def plot_freq_range(df: pd.DataFrame, save: bool = False, verbose: bool = False) -> None:
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

    circuits = df.Circuit.unique()

    freqs = []
    for circ in circuits:
        freq = df[df.Circuit == circ].freq.values
        if verbose:
            print(f"{circ} has {len(freq)} samples")
        min_list = [np.min(f) for f in freq]
        max_list = [np.max(f) for f in freq]
        range_list = list(zip(min_list, max_list))
        # sort
        range_list.sort()
        unique_ranges = list(dict.fromkeys(range_list))
        count = []
        for low, high in unique_ranges:
            # count tuples with same low and high frequency
            r = (low, high)
            count.append(range_list.count(r))
            if verbose:
                print(f"Range {r} has {count[-1]} samples")

        freq = np.concatenate(freq)
        if verbose:
            print(f"Mean freq: {np.mean(freq):.2f}, std freq: {np.std(freq):.2f}")
        freqs.append(freq)

    fig, ax = plt.subplots()
    # plot the freqeuncy range as vlines scaled by the number of samples in that range for each circuit
    for i, circ in enumerate(circuits):
        for j, r in enumerate(unique_ranges):
            # count tuples with same low and high frequency
            high, low = r
            ax.vlines(
                i + 1 + 0.06 * j,
                low,
                high,
                color="k",
                alpha=0.5,
                linewidth=0.02 * count[j],
            )

    # Plot violin plot
    # ax.violinplot(
    #     freqs,
    #     # positions=circuits,
    #     showmeans=True,
    #     showmedians=True,
    #     showextrema=True,
    # )
    # Set x-axis labels
    ax.set_xticks(range(1, len(circuits) + 1), circuits, rotation=45, ha="right")

    ax.set_yscale("log")
    ax.set_ylabel("Frequency range (Hz)")
    # ax.set_xlabel("Equivalent circuit model")
    fig.tight_layout()

    if save:
        fig.savefig("figures/frequency_ranges.eps")
        fig.savefig("figures/frequency_ranges.pdf")
    fig.show()

    return


def meas_points_print(
    df: pd.DataFrame, bubble_plot: bool = True, verbose: bool = False
) -> None:
    """Scatter the measurement points for each circuit type.

    Parameters
    ----------
    df: DataFrame
        Dataframe with EIS data
    save: bool
        Save figure
    verbose: bool
        Print frequency information
    """
    # Colors from: http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # Accessed: 2023-04-14
    colors = [
        "#999999",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    circuits, _ = np.unique(df.Circuit, return_index=True)
    measurement_dict = {}
    for circ in circuits:
        df_unique_circ = df[df["Circuit"] == circ]
        meas_counts_df = df_unique_circ["freq"].apply(lambda x: len(x))
        unique_meas_counts = np.sort(meas_counts_df.unique())
        meas_cout = []
        for meas_count in unique_meas_counts:
            meas_cout.append(len(meas_counts_df[meas_counts_df == meas_count]))
        measurement_dict[circ] = list(zip(unique_meas_counts, meas_cout))
        if verbose:
            print(f"Unique measurement counts: {unique_meas_counts}")
            print(f"Number of unique measurement counts: {len(unique_meas_counts)}")

    if bubble_plot:
        fig, ax = plt.subplots()
        for circ in circuits:
            for i in range(len(measurement_dict[circ])):
                ax.scatter(
                    x=circ,
                    y=measurement_dict[circ][i][0],
                    s=5 * measurement_dict[circ][i][1],
                    c=colors[np.mod(i, len(colors))],
                    edgecolors="black",
                    linewidth=1,
                )
        # Add empty legend box with custom text
        # tilt x labels
        ax.set_xticklabels(circuits, rotation=45, ha="right")
        ax.set_ylabel("Measurement count")
        #  ax.set_xlabel("Equivalent circuit model")
        fig.tight_layout()
        # ax.set_title("Number of spectra with number of measurements and circuit type")
        fig.savefig("figures/measurement_count.pdf")
        fig.savefig("figures/measurement_count.eps")

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
    for i, ind in enumerate(df_sorted.index):
        d = df_sorted.loc[ind]
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

    # x = np.linspace(0.0, 1.0, 9)
    # colors = cm.get_cmap("tab10")(x)[np.newaxis, :, :3]

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


def plot_all_spectra(
    df_sorted: pd.DataFrame,
    fig: Optional[plt.figure] = None,
    ax: Optional[plt.axes] = None,
    plot_real: bool = True,
    save: bool = False,
    color: str = "k",
    return_fig: bool = False,
):
    """Plot all the spectra in the dataframe"""
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(df_sorted)):
        if plot_real:
            ax.plot(df_sorted["f"][i], df_sorted["zreal"][i], color=color, alpha=0.6)
        else:
            ax.plot(df_sorted["f"][i], df_sorted["zimag"][i], color=color, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    if plot_real:
        ax.set_ylabel("Real(Z)")
    else:
        ax.set_ylabel("Imag(Z)")
    ax.set_title("All Spectra")
    if save:
        fig.savefig("figures/all_spectra.png", dpi=300)

    if return_fig:
        return fig, ax
    else:
        plt.show()


def plot_all_nyquist(
    df: pd.DataFrame,
    ax: Optional[plt.axes] = None,
    x_name: str = "",
    y_name: str = "",
    title: str = "",
    linewidth: int = 3,
    save_name: str = "",
    alpha: float = 0.4,
    labels_on: bool = True,
    label_fontsize: int = 12,
    drop_below_zero: bool = False,
) -> None:
    """Plot all the spectra in the dataframe nyquist"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    for i in df.index:
        x = df[x_name].loc[i].copy()
        y = -df[y_name].loc[i].copy()
        if drop_below_zero:
            mask = y > 0
            x = x[mask]
            y = y[mask]
        ax.plot(
            x,
            y,
            linewidth=linewidth,
            color="black",
            alpha=alpha,
        )
    if labels_on:
        ax.set_xlabel(r"$\operatorname{Re}(\tilde{Z})$", fontsize=label_fontsize)
        ax.set_ylabel(r"$\operatorname{Im}(\tilde{Z})$", fontsize=label_fontsize)
        ax.set_title(title, fontsize=label_fontsize + 2)

    if save_name != "":
        fig.savefig(f"figures/{save_name}.pdf")

    return ax


def visualize_raw_spectra(
    df,
    show=1,
    save_figs=0,
    row_col_ratio=0.6,
    pdf=True,
    fig_name="eis_art_",
    sup_title="EIS Spectra",
    axis_off=True,
    label_fontsize=12,
):
    # Visualize data as images
    # Calculate the number of rows and columns bnased on the row_col_ratio.
    rows_df = df.shape[0]

    rows = int(np.ceil(np.sqrt(rows_df * row_col_ratio)))
    cols = int(np.ceil(np.sqrt(rows_df / row_col_ratio)))
    print(
        f"Generating an EIS plot of {rows_df} spectra with {rows} rows and {cols} columns."
    )

    fig, axs = plt.subplots(
        rows,
        cols,
        sharex=False,
        sharey=False,
        figsize=(3.8 * cols, 3 * rows),
        frameon=False,
    )
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                axs[i, j].set_ylabel("-zimag", fontsize=label_fontsize)
            if axis_off:
                axs[i, j].set_axis_off()
            if i * cols + j < rows_df:
                axs[i, j].plot(
                    df.loc[df.index[i * cols + j]]["zreal"],
                    -df.loc[df.index[i * cols + j]]["zimag"],
                    linewidth=1,
                    color="black",
                )
                # Scatter plot of the datapoints as blue crosses.
                axs[i, j].scatter(
                    df.loc[df.index[i * cols + j]]["zreal"],
                    -df.loc[df.index[i * cols + j]]["zimag"],
                    marker="x",
                    s=15,
                    color="blue",
                )
                # Insert thin black lines for the x and y axes.
                axs[i, j].axhline(y=0, color="black", linewidth=0.5)
                axs[i, j].axvline(x=0, color="black", linewidth=0.5)

                if i * cols + j >= rows_df - cols:
                    axs[i, j].set_xlabel("zreal", fontsize=label_fontsize)
            else:
                axs[i, j].set_visible(False)

    # Add a title to the figure.
    fig.suptitle(sup_title, fontsize=6 * cols)
    fig.tight_layout()
    if save_figs:
        fig.savefig(f"figures/{fig_name}.pdf")
        fig.savefig(f"figures/{fig_name}.eps")

    if show:
        plt.show()
    return fig


def confusion_nyquist_plot(
    df: pd.DataFrame,
    le: LabelEncoder,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    columns: list = ["zreal_norm", "zimag_norm"],
    lw: float = 0.2,
    alpha: float = 0.6,
    save: bool = True,
    figname: str = "confusion_nyquist",
    drop_below_zero: bool = False,
    show=True,
):
    circuits = le.inverse_transform(list(range(9)))
    nrow = len(circuits)
    ncol = len(circuits)

    nb_classes = len(le.classes_)
    cm_index_df = pd.DataFrame(index=le.classes_, columns=le.classes_)

    for i in range(nb_classes):
        for j in range(nb_classes):
            cm_index_df.iloc[i, j] = np.where((y_test == i) & (y_test_pred == j))[0]

    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    # Inspired by and snippets from:
    # https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    gs = gridspec.GridSpec(
        nrow,
        ncol,
        wspace=0.02,
        hspace=0.02,
        top=1.0 - 0.5 / (nrow + 1),
        bottom=0.5 / (nrow + 1),
        left=0.5 / (ncol + 1),
        right=1 - 0.5 / (ncol + 1),
    )

    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            # Call function to plot spectra (:
            df_cm_entry = df.iloc[cm_index_df.iloc[i, j]].copy()
            ax = plot_all_nyquist(
                df_cm_entry,
                ax,
                columns[0],
                columns[1],
                title="",
                linewidth=lw,
                alpha=alpha,
                labels_on=False,
                drop_below_zero=drop_below_zero,
            )

            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
            if i == nrow - 1:
                # rotate the lables 45 degrees
                ax.set_xlabel(circuits[j], fontsize=6)  # , rotation = 90)
            if j == 0:
                ax.set_ylabel(circuits[i], fontsize=6)
    fig.text(0.5, 0.01, "Predicted ECM", ha="center")
    fig.text(0.01, 0.5, "Given ECM", va="center", rotation="vertical")
    # Save
    if save:
        fig.savefig(f"figures/{figname}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/{figname}.eps", bbox_inches="tight")
    if show:
        plt.show()
    else:
        return fig, ax
