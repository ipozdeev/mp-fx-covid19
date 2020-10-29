import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
palette = sns.color_palette("colorblind", n_colors=5)

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = "serif"
plt.rcParams['figure.figsize'] = (8.27 - 2, 11.69 / 2.5)


def describe_events(events_df) -> str:
    """Describe events.

    Parameters
    ----------
    events_df : pandas.DataFrame

    Returns
    -------
    output : str
        HTML code to generate a table
    """
    # add total count
    output = events_df.copy()
    output.loc["total no."] = events_df.count()
    output.loc["total cut"] = events_df.sum()

    # rename columns, format dates, report in bps
    def time_fmt(x):
        return x if isinstance(x, str) else x.strftime("%m/%d %H:%M")

    def float_fmt(x):
        return "" if np.isnan(x) else "{:.0f}".format(x)

    output = output \
        .reset_index() \
        .to_html(formatters={"time": time_fmt},
                 float_format=float_fmt,
                 index=False, index_names=False,
                 notebook=True)

    return output


def availability_map(df):
    """Plots

    Parameters
    ----------
    df

    Returns
    -------

    """
    available = df.mul(np.nan)
    ar = np.arange(df.shape[1]) + 1
    available.iloc[0] = ar
    available = available.ffill().where(df.notnull())

    fig, ax = plt.subplots()
    available.plot(ax=ax,
                   color=np.resize(np.array(palette), (df.shape[1], 3)))
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax.set_yticks(ar[::2])
    ax2.set_yticks(ar[1::2])
    ax.set_yticklabels(df.columns[::2])
    ax2.set_yticklabels(df.columns[1::2])
    ax.legend_.remove()

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_xlabel("", visible=False)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # fmt = mdates.DateFormatter('%m/%d')
    # ax.xaxis.set_major_formatter(fmt)

    return fig, ax


if __name__ == '__main__':
    availability_map(pd.DataFrame(np.random.normal(size=(20, 20))))
