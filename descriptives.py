import pandas as pd
import numpy as np


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
