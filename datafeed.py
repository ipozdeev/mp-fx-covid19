import pandas as pd
from joblib import Memory

# from foolbox.data_utilities import parse_bloomberg_excel

cachedir = "/home/ipozdeev/projects/covid-fx/data"
memory = Memory(cachedir, verbose=0)


@memory.cache
def get_fx_data_bloomberg() -> pd.DataFrame:
    """Load spot FX data.

    I downloaded data from Bloomberg in two steps. This function parses the
    files and returns a combined dataset.
    Quotes in the Excel files below are snapped at Zurich time (UTC+1).

    """
    # meta
    meta = pd.read_excel("data/fx_data_intraday.xlsx", sheet_name="iso",
                         index_col=0, header=0)

    data = parse_bloomberg_excel("data/fx_data_intraday.xlsx",
                                 data_sheets="spot",
                                 colnames_sheet="colnames",
                                 skiprows=4)

    data_pt2 = parse_bloomberg_excel("data/fx_data_intraday_pt2.xlsx",
                                     data_sheets="spot",
                                     colnames_sheet="colnames",
                                     skiprows=4)

    data = {1: data, 2: data_pt2}

    # direct/indirect quotes
    for k, v in data.items():
        # flip USDXXX quotes
        data[k].loc[:, meta["usdxxx"].astype(bool)] =\
            data[k].loc[:, meta["usdxxx"].astype(bool)].pow(-1)

        # Bloomberg does not report quotes at the end of the interval
        data[k].index = data[k].index\
            .to_period(freq="15T").end_time.ceil("15T")\
            .tz_localize("Europe/Zurich")

        data[k].columns = data[k].columns.str.lower()

    # merge two parts
    res = data[1].reindex(index=data[1].index.union(data[2].index))
    res.update(data[2])
    res = res.astype(float)

    return res


@memory.cache
def get_fx_data_eikon():
    """Load data from Eikon.

    Data from Eikon is snapped at GMT+1.
    """
    # meta
    meta = pd.read_excel("data/fx_data_intraday_eikon.xlsx", sheet_name="iso",
                         index_col=0, header=0)

    data = parse_bloomberg_excel("data/fx_data_intraday_eikon.xlsx",
                                 data_sheets="data",
                                 colnames_sheet="colnames",
                                 skiprows=3)

    # direct/indirect quotes
    data.loc[:, meta["usdxxx"].astype(bool)] =\
        data.loc[:, meta["usdxxx"].astype(bool)].pow(-1)
    data.index = data.index \
        .tz_localize("Etc/GMT+1").tz_convert("Etc/GMT+2")\
        .tz_localize(None).tz_localize("Europe/Zurich")
    data.columns = data.columns.str.lower()

    data = data.astype(float)

    return data


def get_fx_data() -> pd.DataFrame:
    """Load spot FX quotes, merged Eikon and Bloomberg.

    Returns
    -------
    pandas.DataFrame
        indexed by datetime instances (Zurich time), columned by iso-3 of
        currencies

    """
    # Eikon: 5-min data, needs resampling to 15-min
    data_e = get_fx_data_eikon()
    data_e = data_e.resample("15T", closed="right", label="right").last()

    # Bloomberg
    data_b = get_fx_data_bloomberg()

    # merge
    res = data_e.reindex(index=data_e.index.union(data_b.index),
                         columns=data_e.columns.union(data_b.columns))
    res.fillna(data_b, inplace=True)

    res = res.rename_axis(columns="currency", index="time")

    return res


@memory.cache
def get_stock_data():
    """
    """
    # other -----------------------------------------------------------------
    # meta
    meta_other = pd.read_excel("data/stock-intraday.xlsx",
                               sheet_name="meta_other",
                               header=0)\
        .T.squeeze().str.lower()

    data = parse_bloomberg_excel("data/stock-intraday.xlsx",
                                 data_sheets="other",
                                 colnames_sheet="meta_other",
                                 skiprows=3)
    data = data.rename(columns=meta_other) \
        .resample("15T").last()\
        .dropna(how="all")

    data.index = data.index \
        .to_period(freq="15T").end_time.ceil("15T") \
        .tz_localize("Europe/Zurich", ambiguous="NaT")

    data_other = data.astype(float)

    # msci ------------------------------------------------------------------
    meta_msci = pd.read_excel("data/stock-intraday.xlsx",
                              sheet_name="meta_msci",
                              header=0).T
    meta_msci = meta_msci.loc[meta_msci[0].eq(meta_msci[1])]

    data = parse_bloomberg_excel("data/stock-intraday.xlsx",
                                 data_sheets="msci",
                                 colnames_sheet="meta_msci",
                                 skiprows=3)
    data = data\
        .reindex(columns=meta_msci.index)\
        .rename(columns=meta_msci[0].str.lower()) \
        .resample("15T").last()\
        .dropna(how="all")

    data.index = data.index \
        .to_period(freq="15T").end_time.ceil("15T") \
        .tz_localize("Europe/Zurich", ambiguous="NaT")

    data_msci = data.astype(float)

    res = dict(msci=data_msci, other=data_other)

    return res


def get_events_data() -> pd.DataFrame:
    """Retrieve events, indexed by datetime and columned by iso-3."""
    # meta
    data = pd.read_excel(
        "data/measures.ods", sheet_name=None, engine="odf",
        index_col=0, header=0, parse_dates=True,
        date_parser=lambda dt: pd.to_datetime(dt).tz_localize("Europe/Zurich")
    )

    for k, v in data.items():
        data[k] = v.loc[v["comment"].str.contains(" cut"), "comment"]\
            .rename(k)
        data[k] = data[k].str.extract("([0-9]+) ?bps", expand=False)\
            .astype(float) * -1

    res = pd.concat(data, axis=1).rename_axis(index="time",
                                              columns="currency")\
        .dropna(axis=1, how="all")

    return res


if __name__ == '__main__':
    # get_fx_data_eikon()
    # get_stock_data()
    get_fx_data()
