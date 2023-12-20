"""
Utility functions for the "Employee Retention Project".
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

# Standard library modules
import pathlib
import pickle
from typing import Any

# Third-party modules
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from titlecase import titlecase
from typing_extensions import Self
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------

__author__ = "Markku Laine"
__email__ = "markku.laine@gmail.com"


# ----------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A custom transformer for selecting specific features from a pandas dataframe.

    Args:
        BaseEstimator: A base class for all estimators in scikit-learn.
        TransformerMixin: A mixin class for all transformers in scikit-learn.
    """

    def __init__(self, feature_names: list[str]) -> None:
        """
        Initializes the instance based on given feature names.

        Args:
            feature_names (list[str]): A list of feature names to be selected.
        """
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Self:
        """
        Fits the transformer to a given data.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series, optional): The target variable (defaults to None).

        Returns:
            FeatureSelector: The instance of the FeatureSelector transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selects specified features from a given feature matrix.

        Args:
            X (pd.DataFrame): The feature matrix.

        Returns:
            pd.DataFrame: The Pandas dataframe containing only the selected features.
        """
        return X[self.feature_names]


# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------


def show_shape_and_size(df: pd.DataFrame) -> None:
    """
    Displays the shape and size information of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe to be analyzed.
    """
    rows, columns = df.shape
    elements = df.size
    print(f"Shape: {rows:,} rows and {columns:,} columns")
    print(f"Size:  {elements:,} elements")


def show_column_names(df: pd.DataFrame) -> None:
    """
    Displays the column names of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe whose column names are to be shown.
    """
    columns = df.columns.tolist()
    print("Columns: {}".format(", ".join(columns)))


def show_unique_values(df: pd.DataFrame, column_name: str) -> None:
    """
    Displays the unique values in a specified column of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose unique values are to be shown.
    """
    unique_values = pd.unique(df[column_name])
    print("Unique values in '{}': {}".format(column_name, ", ".join(sorted(unique_values, key=lambda x: x.lower()))))


def show_categories(df: pd.DataFrame, column_name: str) -> None:
    """
    Displays the categories and ordering information for a specified column in a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose categories and ordering information are to be shown.
    """
    print(
        "Categories in '{}' (ordered={}): {}".format(
            column_name, df[column_name].cat.ordered, ", ".join(df[column_name].cat.categories)
        )
    )


def show_dist_values(df: pd.DataFrame, column_name: str) -> None:
    """
    Displays the distribution of values in a specified column of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose distribution of values are to be shown.
    """
    values_count = df[column_name].value_counts(normalize=False).sort_index()
    values_percentage = round(df[column_name].value_counts(normalize=True).sort_index() * 100, 1)
    display(pd.DataFrame(data={"Count (#)": values_count, "Percentage (%)": values_percentage}))


def show_dist_values_grouped(df: pd.DataFrame, column_name: str) -> None:
    """
    Displays the distribution of grouped values in a specified column of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose distribution of grouped values are to be shown.
    """
    # Group 1
    values_count_1 = df.groupby([column_name], observed=False)["left"].value_counts(normalize=False).sort_index()
    values_percentage_1 = round(
        df.groupby([column_name], observed=False)["left"].value_counts(normalize=True).sort_index() * 100, 1
    )
    display(pd.DataFrame(data={"Count (#)": values_count_1, "Percentage (%)": values_percentage_1}))

    # Group 2
    values_count_2 = df.groupby(["left"], observed=False)[column_name].value_counts(normalize=False).sort_index()
    values_percentage_2 = round(
        df.groupby(["left"], observed=False)[column_name].value_counts(normalize=True).sort_index() * 100, 1
    )
    display(pd.DataFrame(data={"Count (#)": values_count_2, "Percentage (%)": values_percentage_2}))


def show_mean_median_grouped(df: pd.DataFrame, column_name: str) -> None:
    """
    Displays the mean and median values of a specified column in a given pandas dataframe, grouped by the "left" variable.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose grouped mean and median values are to be shown.
    """
    display(df.groupby("left")[column_name].agg(["mean", "median"]).rename(columns={"mean": "Mean", "median": "Median"}))


def check_missing_values(df: pd.DataFrame) -> None:
    """
    Checks and displays information about missing values in a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe to be checked for missing values.
    """
    missing_values_count = df.isna().sum()
    missing_values_percentage = round(df.isna().mean() * 100, 2)
    display(pd.DataFrame(data={"Missing values (#)": missing_values_count, "Missing values (%)": missing_values_percentage}))


def check_duplicates(df: pd.DataFrame) -> None:
    """
    Checks and displays information about duplicate rows in a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe to be checked for duplicate rows.
    """
    duplicates_count = df.duplicated().sum()
    duplicates_percentage = df.duplicated().mean()
    print(f"Duplicates: {duplicates_count:,} ({duplicates_percentage:.2%})")


def check_outliers(df: pd.DataFrame, column_names: list[str]) -> None:
    """
    Checks and displays information about outliers in specified columns of a given pandas dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe to be checked for outliers.
        column_names (list[str]): A list of column names whose outliers are to be checked.
    """
    outliers_counts = []
    outliers_percentages = []
    for column_name in column_names:
        # Use the IQR method
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - iqr * 1.5
        upper_limit = q3 + iqr * 1.5
        outliers_mask = (df[column_name] < lower_limit) | (upper_limit < df[column_name])
        outliers_count = df[outliers_mask].shape[0]
        outliers_percentage = round(outliers_count / df.shape[0] * 100, 2)
        outliers_counts.append(outliers_count)
        outliers_percentages.append(outliers_percentage)

    display(pd.DataFrame(data={"Outliers (#)": outliers_counts, "Outliers (%)": outliers_percentages}, index=column_names))


def visualize_outliers(
    df: pd.DataFrame,
    column_name: str,
    width: int = 10,
    height: int = 5,
    box_color: str = "#E7E7E7",
    flier_color: str = "#FF8A80",
) -> None:
    """
    Visualizes outliers in a specified column of a given pandas dataframe using a boxplot.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose outliers are to be visualized.
        width (int, optional): The width of the plot (defaults to 10).
        height (int, optional): The height of the plot (defaults to 5).
        box_color (str, optional): The color of the box in the boxplot (defaults to "#E7E7E7").
        flier_color (str, optional): The color of the outlier points in the boxplot (defaults to "#FF8A80").
    """
    # Create a subplot
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(
        "Boxplot Analysis: Uncovering Outliers in {}".format(titlecase(column_name.replace("_", " "))),
        fontsize=14,
        weight="normal",
        y=0.8,
    )

    # ----- PLOT 1 -----
    # Create a boxplot
    sns.boxplot(
        ax=ax,
        data=df,
        x=column_name,
        width=0.5,
        color=box_color,
        showfliers=True,
        flierprops=dict(markerfacecolor=flier_color, markersize=6, markeredgecolor="none"),
    )

    # Customize axis spines
    sns.despine(offset=-40, trim=True)

    # Customize ticks
    ax.tick_params(axis="x", colors="#595959")  # set x-axis tick color
    ax.set_xticklabels(ax.get_xticklabels(), color="#595959", ha="center")  # set x-axis tick labels
    ax.set_yticks([])  # remove y-axis ticks

    # Customize labels
    ax.set_xlabel(titlecase(column_name.replace("_", " ")), color="#595959")  # set x-axis label
    ax.set_ylabel("")  # remove y-axis label

    # Show the plot
    plt.show()


def visualize_dist_cat(
    df: pd.DataFrame,
    column_name: str,
    label_dict: dict[str | int, str] | None = None,
    rotation: int = 0,
    width: int = 10,
    width_ratios: list[int] = [1, 2],
    height: int = 4,
    colors: list[str] = sns.color_palette("Paired").as_hex(),
    value_distance: float = 0.6,
    value_colors: list[str] = ["#595959", "#FFFFFF"],
    value_size: int = 10,
    explodes: list[float] | None = None,
    counterclock: bool = True,
    sort_alphabetically: bool = True,
) -> None:
    """
    Visualizes the distribution of employees in a specified categorical column in a given pandas dataframe using a countplot and a pie chart.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the categorical column whose distribution of employees is to be visualized.
        label_dict (dict[str | int, str] | None, optional): A dictionary for customizing tick labels (defaults to None).
        rotation (int, optional): The rotation angle for x-axis tick labels (defaults to 0).
        width (int, optional): The width of the plot (defaults to 10).
        width_ratios (list[int], optional): The width ratios of the subplots (defaults to [1, 2]).
        height (int, optional): The height of the plot (defaults to 4).
        colors (list[str], optional): A list of category colors for the countplot and pie chart (defaults to Seaborn's
            "Paired" color palette).
        value_distance (float, optional): The relative distance along the radius at which the text generated by autopct is
            drawn (defaults to 0.6).
        value_colors (list[str], optional): The colors for count values on top of each bar and percentage values in the
            pie chart (defaults to ["#595959", "#FFFFFF"]).
        value_size (int, optional): The font size for count values on top of each bar and percentage values in the pie
            chart (defaults to 10).
        explodes (list[float] | None, optional): A list of explosion distances for each wedge in the pie chart (defaults to
            None).
        counterclock (bool, optional): Whether to arrange pie chart fractions counterclockwise (defaults to True).
        sort_alphabetically (bool, optional): Whether to sort categories alphabetically (defaults to True).
    """
    # Define helper variables
    plot_color_1 = "#595959"
    plot_color_2 = "#E7E7E7"
    value_counts_s = df[column_name].value_counts()
    if sort_alphabetically:  # incl. a hack for the bug in .sort_index()
        value_counts_l = sorted(
            list(zip(value_counts_s.index, value_counts_s))
        )  # convert values & counts to a list of tuples, sorted alphabetically by value
    else:
        value_counts_l = list(zip(value_counts_s.index, value_counts_s))  # convert values & counts to a list of tuples
    labels = [t[0] for t in value_counts_l]
    sizes = [t[1] for t in value_counts_l]
    order = labels
    n_categories = len(order)
    explodes = explodes if explodes is not None else [0] * n_categories
    if len(colors) < n_categories:
        colors = [colors[i % len(colors)] for i in range(n_categories)]
    else:
        colors = colors[:n_categories]
    palette = {value: color for value, color in zip(labels, colors)}

    # Create a figure and subplots
    fig, ax = plt.subplots(1, 2, width_ratios=width_ratios, figsize=(width, height))  # 1 x 2 subplots (side by side)
    fig.suptitle(
        "Distribution of Employees by {}".format(titlecase(column_name.replace("_", " "))),
        fontsize=14,
        weight="normal",
        y=1.0,
    )

    # ----- PLOT 1 -----
    # Create a countplot
    sns.countplot(
        ax=ax[0],
        data=df,
        x=column_name,
        palette=palette,
        hue=column_name,
        order=order,
        stat="count",
        saturation=1,
        alpha=1.0,
        legend=False,
    )

    # Add count values on top of each bar
    for p in ax[0].patches:
        ax[0].annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            color=value_colors[0],
            fontsize=value_size,
            weight="bold",
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Customize axis spines
    sns.despine(trim=False)  # remove axis spines
    ax[0].spines[["top", "right", "left"]].set_visible(False)  # remove some borders
    ax[0].spines[["bottom"]].set_color(plot_color_2)  # set bottom border color
    ax[0].spines[["bottom"]].set_linewidth(1)  # set bottom border width

    # Customize ticks
    ax[0].tick_params(
        axis="x", which="both", bottom=False, top=False, colors=plot_color_1, labelcolor=plot_color_1, rotation=rotation
    )  # set x-axis ticks
    ax[0].tick_params(
        axis="y", which="both", left=False, right=False, colors=plot_color_1, labelcolor=plot_color_1
    )  # set y-axis ticks

    # Customize tick labels
    if label_dict is not None:
        label_dict = {str(k): v for k, v in label_dict.items()}  # convert keys to string
        labels = [
            label_dict[label.get_text()] if label_dict else label.get_text() for label in ax[0].get_xticklabels()
        ]  # get updated x-axis tick labels
        ax[0].set_xticks(range(len(labels)))  # set the number of x-axis tick labels
        ax[0].set_xticklabels(labels, ha="center")  # set x-axis tick labels
    # ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "{:,.0f}".format(x)))
    ax[0].set_yticklabels([])  # remove y-axis tick labels

    # Customize labels
    ax[0].set_xlabel("", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # remove x-axis label
    ax[0].set_ylabel("", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # remove y-axis label

    # ----- PLOT 2 -----
    # Create a pie chart
    wedges, texts, autotexts = ax[1].pie(
        x=sizes,
        labels=labels,
        colors=colors,
        explode=explodes,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        autopct="%1.1f%%",
        pctdistance=value_distance,
        startangle=90,
        counterclock=counterclock,
    )

    # Customize labels and values
    plt.setp(texts, color=plot_color_1)  # set labels
    plt.setp(autotexts, size=value_size, weight="bold", color=value_colors[1])  # set values

    # Show the plots
    plt.show()


def visualize_dist_num(
    df: pd.DataFrame,
    column_name: str,
    discrete: bool = True,
    width: int = 8,
    height: int = 4,
    legend_loc: str = "best",
    colors: list[str] = ["#E7E7E7", "#C7C7C7"],
) -> None:
    """
    Visualizes the distribution of employees in a specified numerical column of a given pandas dataframe using a histogram plot.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the numerical column whose distribution of employees is to be visualized.
        discrete (bool, optional): Whether the data in the specified column is discrete (defaults to True).
        width (int, optional): The width of the plot (defaults to 8).
        height (int, optional): The height of the plot (defaults to 4).
        legend_loc (str, optional): The location of the legend (defaults to "best").
        colors (list[str], optional): A list of colors for the plot elements (defaults to ["#E7E7E7", "#C7C7C7"]).
    """
    # Define helper variables
    colors = colors + sns.color_palette("Paired").as_hex()
    plot_color_1 = "#595959"
    plot_color_2 = "#F0F0F0"
    rotation = 0
    kde = not discrete

    # Create a figure and subplots
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(
        "Distribution of Employees by {}".format(titlecase(column_name.replace("_", " "))),
        fontsize=14,
        weight="normal",
        y=1.0,
    )

    # ----- HIST PLOT -----
    # Create a histogram plot
    sns.histplot(
        ax=ax,
        data=df,
        x=column_name,
        color=colors[0],
        kde=kde,
        shrink=0.8,
        ec="none",
        discrete=discrete,
        stat="frequency",
        alpha=1.0,
    )
    if kde:
        ax.lines[0].set_color(colors[1])  # set kde color
        ax.lines[0].set_label("KDE")  # set kde label
        ax.lines[0].set_linewidth(2)  # set kde line width
    mean = df[column_name].mean()
    median = df[column_name].median()
    ax.axvline(mean, color=colors[3], linestyle="--", linewidth=2, label=f"Mean: {mean:,.2f}")
    ax.axvline(median, color=colors[7], linestyle=":", linewidth=2, label=f"Median: {median:,.2f}")

    # Customize axis spines
    sns.despine(trim=False)  # remove axis spines
    ax.spines[["top", "right", "left"]].set_visible(False)  # remove some borders
    ax.spines[["bottom"]].set_color(plot_color_1)  # set bottom border color
    ax.spines[["bottom"]].set_linewidth(1)  # set bottom border width

    # Customize ticks
    ax.tick_params(
        axis="x", which="both", bottom=True, top=False, colors=plot_color_1, labelcolor=plot_color_1, rotation=rotation
    )  # set x-axis ticks
    ax.tick_params(
        axis="y", which="both", left=False, right=False, colors=plot_color_1, labelcolor=plot_color_1
    )  # set y-axis ticks

    # Customize tick labels
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "{:,.0f}".format(x)))

    # Customize grid lines
    ax.yaxis.grid(color=plot_color_2)  # set y-axis grid line colors
    ax.set_axisbelow(True)  # show axes below

    # Customize labels
    ax.set_xlabel(
        titlecase(column_name.replace("_", " ")), color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal"
    )  # set x-axis label
    ax.set_ylabel(
        titlecase(ax.get_ylabel()), color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal"
    )  # set y-axis label

    # Create legend
    ax.legend(borderpad=0.8, loc=legend_loc, fontsize=8)

    # Show the plot
    plt.show()


def visualize_dist_vs(
    df: pd.DataFrame,
    column_name: str,
    label_dict: dict[str | int, str] | None = None,
    discrete: bool = True,
    rotation: int = 0,
    width: int = 8,
    height: int = 4,
    legend_loc: str = "best",
    legend_dict: dict[str | int, str] = {0: "Stayed", 1: "Left"},
    colors: list[str] = ["#E7E7E7", "#FF8A80"],
) -> None:
    """
    Visualizes the distribution of employees in a categorical or numerical column of a given pandas dataframe with respect to the "left" (target) variable.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the specified column.
        column_name (str): The name of the column whose distribution of employees is to be visualized.
        label_dict (dict[str | int, str] | None, optional): A dictionary for customizing tick labels (defaults to None).
        discrete (bool, optional): Whether the data in the specified column is discrete (defaults to True).
        rotation (int, optional): The rotation angle for x-axis tick labels (defaults to 0).
        width (int, optional): The width of the plot (defaults to 8).
        height (int, optional): The height of the plot (defaults to 4).
        legend_loc (str, optional): The location of the legend (defaults to "best").
        legend_dict (_type_, optional): A dictionary for customizing legend labels (defaults to {0: "Stayed", 1: "Left"}).
        colors (list[str], optional): A list of colors for the plot elements (defaults to ["#E7E7E7", "#FF8A80"]).
    """
    # Define helper variables
    plot_color_1 = "#595959"
    plot_color_2 = "#F0F0F0"

    # Create a figure and subplots
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(
        "Distribution of Employees by {}: Stayed vs. Left".format(titlecase(column_name.replace("_", " "))),
        fontsize=14,
        weight="normal",
        y=1.0,
    )

    # ----- PLOT 1 -----
    if discrete:
        # Create a countplot
        sns.countplot(
            ax=ax,
            data=df,
            x=column_name,
            palette=colors,
            hue="left",
            hue_order=legend_dict.keys(),
            stat="count",
            saturation=1,
            alpha=1.0,
        )
    else:
        # Create a kdeplot
        sns.kdeplot(ax=ax, data=df, x=column_name, palette=colors, hue="left", hue_order=legend_dict.keys(), fill=True)

    # Customize axis spines
    sns.despine(trim=False)  # remove axis spines
    ax.spines[["top", "right", "left"]].set_visible(False)  # remove some borders
    ax.spines[["bottom"]].set_color(plot_color_1)  # set bottom border color
    ax.spines[["bottom"]].set_linewidth(1)  # set bottom border width

    # Customize ticks
    ax.tick_params(
        axis="x", which="both", bottom=True, top=False, colors=plot_color_1, labelcolor=plot_color_1, rotation=rotation
    )  # set x-axis ticks
    ax.tick_params(
        axis="y", which="both", left=False, right=False, colors=plot_color_1, labelcolor=plot_color_1
    )  # set y-axis ticks

    # Customize tick labels
    if label_dict is not None:
        label_dict = {str(k): v for k, v in label_dict.items()}  # convert keys to string
        labels = [
            label_dict[label.get_text()] if label_dict else label.get_text() for label in ax.get_xticklabels()
        ]  # get updated x-axis tick labels
        ax.set_xticks(range(len(labels)))  # set the number of x-axis tick labels
        ax.set_xticklabels(labels, ha="center")  # set x-axis tick labels
    if discrete:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "{:,.0f}".format(x)))

    # Customize grid lines
    ax.yaxis.grid(color=plot_color_2)  # set y-axis grid line colors
    ax.set_axisbelow(True)  # show axes below

    # Customize labels
    ax.set_xlabel(
        titlecase(column_name.replace("_", " ")), color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal"
    )  # set x-axis label
    ax.set_ylabel(
        titlecase(ax.get_ylabel()), color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal"
    )  # set y-axis label

    # Create legend
    legend_labels = legend_dict.values()  # get custom labels for the legend
    legend_lines = [
        Line2D([0], [0], color=color, lw=2) for color, label in zip(colors, legend_labels)
    ]  # create custom objects for the legend
    ax.legend(handles=legend_lines, labels=legend_labels, borderpad=0.8, loc=legend_loc, fontsize=8)

    # Show the plot
    plt.show()


def visualize_corr_hm(
    df: pd.DataFrame,
    drop_columns: list[str] = [],
    filter_column: str | None = None,
    filter_value: float | None = None,
    subtitle: str | None = None,
    width: int = 8,
    height: int = 6,
) -> None:
    """
    Visualizes the correlation heatmap of a given pandas dataframe, optionally dropping columns and applying filters.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the data.
        drop_columns (list[str], optional): A list of column names to be dropped from the correlation analysis (defaults to
            an empty list).
        filter_column (str | None, optional): The column name for applying a filter (defaults to None).
        filter_value (float | None, optional): The value used for filtering a specified column (defaults to None).
        subtitle (str | None, optional): An optional subtitle to be appended to the title of the heatmap (defaults to None).
        width (int, optional): The width of the plot (defaults to 8).
        height (int, optional): The height of the plot (defaults to 6).
    """
    # Copy the dataframe
    df_corr = df.copy()

    # Drop columns
    if drop_columns:
        df_corr = df_corr.drop(columns=drop_columns)

    if "salary_level" not in drop_columns:
        # Encode salary level (order matters)
        salary_level_dict = {
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        df_corr["salary_level"] = df_corr["salary_level"].map(salary_level_dict)

    if "department" not in drop_columns:
        # Encode department (order doesn't matter)
        df_dept = pd.get_dummies(df_corr["department"], prefix="department")
        df_corr = pd.merge(df_corr, df_dept, left_index=True, right_index=True)
        df_corr = df_corr.drop(columns="department")

    # Filter data
    if filter_column is not None and filter_value is not None:
        df_corr = df_corr[df_corr[filter_column] == filter_value].drop(columns=filter_column)

    # Generate a correlation matrix
    correlation_matrix = df_corr.corr(method="pearson")

    # ----- PLOT 1 -----
    # Create a correlation heatmap
    plt.figure(figsize=(width, height))
    heatmap = sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, cmap="vlag", fmt=".2f")
    title = "Correlation Heatmap"
    if subtitle:
        title = f"{title}: {subtitle}"
    heatmap.set_title(title, fontdict={"fontsize": 14}, pad=20)

    # Show the plot
    plt.show()


def visualize_corr_pr(df: pd.DataFrame) -> None:
    """
    Visualizes the pairwise relationships between numerical variables in a given pandas dataframe, highlighting employees by
    their "left" status.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the data.
    """
    # Define helper variables
    colors = ["#C7C7C7", "#FF8A80"]

    # Copy the dataframe
    df_corr = df.copy()

    # Sort data to show employees who left on top
    df_corr = df_corr.sort_values(by="left", ascending=True)

    # ----- PLOT 1 -----
    # Create a pairplot
    pairplot = sns.pairplot(df_corr, hue="left", palette=colors, plot_kws={"s": 5, "alpha": 0.5})
    pairplot.fig.suptitle("Pairwise Relationships: Stayed vs. Left", fontsize=14, weight="normal", y=1.04)
    pairplot.fig.set_size_inches(11, 11)

    # Customize legend
    pairplot._legend.remove()  # remove default legend
    legend_dict = {0: "Stayed", 1: "Left"}  # define legend labels
    legend_labels = legend_dict.values()  # get custom labels for the legend
    legend_lines = [
        Line2D([0], [0], color=color, lw=2) for color, label in zip(colors, legend_labels)
    ]  # create custom objects for the legend
    pairplot.fig.legend(
        handles=legend_lines, labels=legend_labels, loc="upper right", borderpad=0.8, fontsize=8, bbox_to_anchor=(0.95, 1.05)
    )

    # Show the plot
    plt.show()


def visualize_feature_importances(
    model: Pipeline, model_name: str, colors: list[str] = ["#E7E7E7"], width: int = 8, height: int = 4
) -> None:
    """
    Visualizes the feature importances of a given model using a horizontal bar plot.

    Args:
        model (Pipeline): The trained model (pipeline).
        model_name (str): The name of the model to be displayed in the plot title.
        colors (list[str], optional): A list of colors for the bar plot (defaults to ["#E7E7E7"]).
        width (int, optional): The width of the plot (defaults to 8).
        height (int, optional): The height of the plot (defaults to 4).
    """
    # Create feature importances dataframe
    feature_names = model.named_steps["clf"].feature_names_in_
    feature_importances = model.named_steps["clf"].feature_importances_
    df_fi = (
        pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Define helper variables
    colors = colors
    plot_color_1 = "#595959"
    plot_color_2 = "#F0F0F0"

    # Create a figure and subplots
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(f"Feature Importances: {model_name}", fontsize=14, weight="normal", y=1.0)

    # ----- PLOT 1 -----
    # Create a bar plot
    sns.barplot(ax=ax, data=df_fi, x="Importance", y="Feature", color=colors[0], ec="none", alpha=1.0)

    # Customize axis spines
    sns.despine(trim=False)  # remove axis spines
    ax.spines[["top", "right", "bottom"]].set_visible(False)  # remove some borders
    ax.spines[["left"]].set_color(plot_color_1)  # set bottom border color
    ax.spines[["left"]].set_linewidth(1)  # set bottom border width

    # Customize ticks
    ax.tick_params(
        axis="x", which="both", bottom=False, top=False, colors=plot_color_1, labelcolor=plot_color_1
    )  # set x-axis ticks
    ax.tick_params(
        axis="y", which="both", left=True, right=False, colors=plot_color_1, labelcolor=plot_color_1
    )  # set y-axis ticks

    # Customize grid lines
    ax.xaxis.grid(color=plot_color_2)  # set y-axis grid line colors
    ax.set_axisbelow(True)  # show axes below

    # Customize labels
    ax.set_xlabel("Importance", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # set x-axis label
    ax.set_ylabel("Feature", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # set y-axis label

    # Show the plot
    plt.show()


def get_best_results(model_name: str, optimizer: GridSearchCV, precision: int = 6) -> pd.DataFrame:
    """
    Retrieves the best classification performance results for a specific model from a given GridSearchCV optimizer.

    Args:
        model_name (str): The name of the model whose best classification performance results are to be obtained.
        optimizer (GridSearchCV): The GridSearchCV object containing the results.
        precision (int, optional): The number of decimal places to round the results (defaults to 6).

    Returns:
        pd.DataFrame: A pandas dataframe summarizing the best classification performance results for a specific model.
    """
    cv_results: pd.DataFrame = pd.DataFrame(optimizer.cv_results_)
    best_results: pd.Series = cv_results.iloc[optimizer.best_index_, :]
    return pd.DataFrame(
        {
            "Model": [model_name],
            "Accuracy": [round(best_results["mean_test_accuracy"], precision)],
            "Precision": [round(best_results["mean_test_precision"], precision)],
            "Recall": [round(best_results["mean_test_recall"], precision)],
            "F1": [round(best_results["mean_test_f1"], precision)],
        }
    )


def get_results(model_name: str, y_true: pd.Series, y_pred: np.ndarray, precision: int = 6) -> pd.DataFrame:
    """
    Computes classification performance results for a specific model.

    Args:
        model_name (str): The name of the model whose classification performance results are to be computed.
        y_true (pd.Series): The ground truth (correct) target values.
        y_pred (np.ndarray): The predicted target values.
        precision (int, optional): The number of decimal places to round the results (defaults to 6).

    Returns:
        pd.DataFrame: A pandas dataframe summarizing the classification performance results for a specific model.
    """
    return pd.DataFrame(
        {
            "Model": [model_name],
            "Accuracy": [round(accuracy_score(y_true, y_pred), precision)],
            "Precision": [round(precision_score(y_true, y_pred, zero_division=np.nan), precision)],
            "Recall": [round(recall_score(y_true, y_pred), precision)],
            "F1": [round(f1_score(y_true, y_pred), precision)],
        }
    )


def get_updated_results(
    results: pd.DataFrame, new_results: list[pd.DataFrame] | None = None, sort_by: list[str] | None = None
) -> pd.DataFrame:
    """
    Updates the existing results with given new results, optionally sorted as specified.

    Args:
        results (pd.DataFrame): The existing pandas dataframe containing results to be updated.
        new_results (list[pd.DataFrame] | None, optional): A list of pandas dataframes containing new results (defaults to
            None).
        sort_by (list[str] | None, optional): A list of column names by which to sort the updated results (defaults to
            None).

    Returns:
        pd.DataFrame: The updated results, sorted as specified.
    """
    new_results = [] if new_results is None else new_results
    sort_by = [] if sort_by is None else sort_by
    return (
        pd.concat([results] + new_results, ignore_index=True).sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    )


def save_model(filename: str, model_object: Pipeline) -> None:
    """
    Saves a given model (pipeline) to a specified file.

    Args:
        filename (str): The name of the file to which the model (pipeline) will be saved.
        model_object (Pipeline): The model (pipeline) object to be saved.
    """
    filepath = pathlib.Path(f"models/{filename}")

    with open(filepath, "wb") as f:
        pickle.dump(model_object, f)


def load_model(filename: str) -> Pipeline:
    """
    Loads a given model (pipeline) from a specified file.

    Args:
        filename (str): The name of the file from which the model (pipeline) will be loaded.

    Returns:
        Pipeline: The loaded model (pipeline) object.
    """
    filepath = pathlib.Path(f"models/{filename}")

    with open(filepath, "rb") as f:
        model_object = pickle.load(f)

    return model_object


def build_tuned_model(
    metadata: dict[str, Any],
    model: DecisionTreeClassifier | RandomForestClassifier | XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    scoring: list[str] | None = None,
    cv: int | None = 5,
    refit: str = "recall",
    show_results: bool = True,
    save: bool = True,
) -> Pipeline:
    """
    Builds a tuned machine learning model (pipeline) using a given classifier.

    Args:
        metadata (dict[str, Any]): The metadata containing information about the model.
        model (DecisionTreeClassifier | RandomForestClassifier | XGBClassifier): The classifier to be used.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        feature_names (list[str] | None, optional): A list of feature names to be selected (defaults to None, uses all
            columns of X).
        scoring (list[str] | None, optional): A list of scoring metrics for cross-validation (defaults to None, uses
            ["accuracy", "precision", "recall", "f1"]).
        cv (int | None, optional): The number of cross-validation folds (defaults to 5).
        refit (str, optional): The scoring metric to use for refitting the model (defaults to "recall").
        show_results (bool, optional): Whether to display model evaluation results (defaults to True).
        save (bool, optional): Whether to save the trained model to a file (defaults to True).

    Returns:
        Pipeline: The pipeline containing the tuned model.
    """
    # Set default values for mutable objects, if needed
    feature_names = X.columns.tolist() if feature_names is None else feature_names
    scoring = ["accuracy", "precision", "recall", "f1"] if scoring is None else scoring

    # Get model metadata
    model_abbreviation = metadata["abbreviation"]
    model_version = metadata["version"]
    model_name = metadata["name"]
    model_name = f"{model_name} v{model_version}"
    model_param_grid = metadata["param_grid"]
    model_filename = f"{model_abbreviation}{model_version}_pipeline_tr_hr_dataset.pkl"

    # Create pipeline
    pipeline = Pipeline([("fs", FeatureSelector(feature_names)), ("clf", model)])

    # Perform cross-validated grid search
    gscv = GridSearchCV(
        estimator=pipeline, param_grid=model_param_grid, scoring=scoring, cv=cv, refit=refit, n_jobs=-1, verbose=1
    )

    # Fit model
    gscv = gscv.fit(X, y)

    # Get best estimator (pipeline)
    best_estimator: Pipeline = gscv.best_estimator_

    if show_results:
        # Get model results
        best_score = gscv.best_score_
        best_params = gscv.best_params_
        best_results = get_best_results(f"{model_name} (Training)", gscv, precision=4)

        # Show best score and best parameters
        print(f"Best score: {best_score}")
        print(f"Best parameters: {best_params}")

        # Display best results
        display(best_results)

        # Visualize feature importances
        visualize_feature_importances(best_estimator, model_name, colors=["#FDBF6F"])

    # Save model (pipeline)
    if save:
        save_model(model_filename, best_estimator)

    return best_estimator


def compare_tuned_models(
    model_metadata_dict: dict[str, dict[str, Any]], X: pd.DataFrame, y: pd.Series, metric: str = "recall"
) -> tuple[Pipeline | None, dict[str, Any] | None]:
    """
    Compares the performance of tuned machine learning models based on a specified metric.

    Args:
        model_metadata_dict (dict[str, dict[str, Any]]): A dictionary containing the metadata for each model.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        metric (str, optional): The metric used to compare models (defaults to "recall").

    Returns:
        tuple[Pipeline | None, dict[str, Any] | None]: A tuple containing the champion model (pipeline) and its metadata.
    """
    best_score = -1
    champion_model: Pipeline | None = None
    champion_model_metadata: dict[str, Any] | None = None
    all_results = pd.DataFrame()
    for model_metadata in model_metadata_dict.values():
        # Get model metadata
        model_abbreviation = model_metadata["abbreviation"]
        model_version = model_metadata["version"]
        model_name = model_metadata["name"]
        model_name = f"{model_name} v{model_version}"
        model_filename = f"{model_abbreviation}{model_version}_pipeline_tr_hr_dataset.pkl"

        # Load model (pipeline)
        model: Pipeline = load_model(model_filename)

        # Predict
        y_pred = model.predict(X)

        # Get results
        results = get_results(f"{model_name} (Validation)", y, y_pred, precision=4)
        score = results.loc[0, metric.capitalize()]
        if score > best_score:
            best_score = score
            champion_model = model
            champion_model_metadata = model_metadata

        # Get updated results
        all_results = get_updated_results(all_results, new_results=[results], sort_by=[metric.capitalize()])

    # Display all results
    display(all_results)

    return (champion_model, champion_model_metadata)


def evaluate_model(metadata: dict[str, Any], model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the performance of a trained machine learning model.

    Args:
        metadata (dict[str, Any]): Metadata containing information about the model.
        model (Pipeline): The trained model (pipeline).
        X_test (pd.DataFrame): The feature matrix.
        y_test (pd.Series): The ground truth (correct) target values.
    """
    # Get model metadata
    model_version = metadata["version"]
    model_name = metadata["name"]
    model_name = f"{model_name} v{model_version}"

    # Predict classes
    y_pred = model.predict(X_test)

    # Get results
    results = get_results(f"{model_name} (Test)", y_test, y_pred, precision=4)

    # Display results
    display(results)

    # Visualize feature importances
    visualize_feature_importances(model, model_name, colors=["#FDBF6F"])

    # Generate array of values for confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(values_format="d", cmap="YlOrBr")
    plt.grid(False)
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, weight="normal", y=1.05)

    # Show the plot
    plt.show()
