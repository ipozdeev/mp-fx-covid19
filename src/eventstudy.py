import pandas as pd
import numpy as np
import warnings


class EventStudy:
    """Event study machinery.

    Event studies with i={1...N} assets and M(i) asset-specific events.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        variable of interest, most commonly (summable) returns
    events : list or pandas.DataFrame or pandas.Series
        of events (if a pandas object, must be indexed with dates)
    window : tuple
        of int (a,d) where each element is a relative (in
        periods) date to explore patterns in `data`, such as (-5, 5)
    event_date_index

    """

    def __init__(self, data, events, window, event_date_index=0):
        # case when some event dates are not in `data`: this will lead to
        #   conflichts on reshaping
        if len(data.index.union(events.dropna(how="all").index)) > len(data):
            raise ValueError("Some event dates are not present in `data`. "
                             "Align `data` and `events` first.")

        if isinstance(data, pd.Series):
            # to a DataFrame
            data = data.to_frame()

            if isinstance(events, pd.Series):
                # events must be an equally-named frame
                events = events.to_frame(data.columns[0])

        else:
            if isinstance(events, pd.Series):
                # the same events for all data columns
                events = pd.concat([events, ] * data.shape[1],
                                   axis=1,
                                   keys=data.columns)
            else:
                if data.shape[1] > events.shape[1]:
                    warnings.warn("`data` and `events` have several "
                                  "different columns which will be removed.")

        events, data = events.align(data, axis=1, join="inner")

        # data types better be float
        self.data = data.astype(float)
        self.events = events.astype(float)
        self.window = window
        self.event_date_index = event_date_index

        # parameters
        self.assets = self.data.columns

    def mark_event_windows(self, excl_ambig=True):
        """Mark windows around each event.

        For each event in `events`, surrounds the corresponding date with
        indexes from the start to the end of `window`. Ambiguous cases
        that can be characterized as both post-event for some event i and
        pre-event for some event (i+1) are dropped.

        Parameters
        ----------
        excl_ambig : bool

        Returns
        -------
        pandas.DataFrame
            boolean, same shape as `data`

        """
        w_s, w_e = self.window

        w_e -= self.event_date_index

        events_reidx = self.events.reindex_like(self.data)
        events_reidx = events_reidx.notnull().replace(False, np.nan)

        # remove asset-date pairs which can be classified as both pre- and
        # post-events
        mask_confusing = \
            events_reidx.bfill(limit=-w_s).fillna(False).astype(bool) & \
            events_reidx.ffill(limit=w_e).fillna(False).astype(bool) & \
            events_reidx.isnull()

        res = events_reidx\
            .bfill(limit=-w_s)\
            .ffill(limit=w_e)\
            .notnull()

        if excl_ambig:
            res = res.mask(mask_confusing, False)

        return res

    def pivot(self):
        """Reshape variable of interest to have event-centric index.

        Parameters
        ----------

        Returns
        -------
        pandas.DataFrame

        """
        w_s, w_e = self.window

        dates_df = pd.DataFrame.from_dict(
            {c: self.data.index for c in self.data.columns},
            orient="columns"
        )
        dates_df.index = self.data.index
        dates_df.columns.name = self.data.columns.name

        evt_dates = dates_df.where(self.events.notnull())
        periods = self.events.mask(self.events.notnull(),
                                   self.event_date_index) \
            .reindex_like(dates_df)

        count_bef = -1
        count_aft = int(self.event_date_index == 1)

        # grow dates like yeast around event dates
        for _p in range(max(np.abs(self.window))):
            if count_bef >= w_s:
                periods = periods.fillna((periods - 1).bfill(limit=1))
                evt_dates = evt_dates.bfill(limit=1)
                count_bef -= 1
            if count_aft <= w_e:
                periods = periods.fillna((periods + 1).ffill(limit=1))
                evt_dates = evt_dates.ffill(limit=1)
                count_aft += 1

        # remove ambiguous cases
        evt_w = self.mark_event_windows(excl_ambig=True)
        evt_dates = evt_dates.where(evt_w)
        periods = periods.where(evt_w)

        # concat
        dt_period_pairs = pd.concat(
            (evt_dates, periods),
            axis=1,
            keys=["evt_dt", "d_periods"]
        ).stack(level=1)

        res = pd.concat(
            (dt_period_pairs,
             self.data.stack().rename("val") \
             .reindex(index=dt_period_pairs.index)),
            axis=1
        )

        # put d_periods on the x-axis, assets/dates on the y-axis
        res = res \
            .set_index(["evt_dt", "d_periods"], append=True) \
            .droplevel(0, axis=0) \
            .squeeze() \
            .unstack(level=[0, 1])

        # delete periods outside the event window
        res = res.loc[w_s:w_e]

        # sort
        res = res.sort_index(axis=0).sort_index(axis=1)

        return res

    @staticmethod
    def event_weighted_mean(df) -> pd.Series:
        """Calculate the mean weighted by the number of events for each asset.

        Parameters
        ----------
        df : pandas.DataFrame
        """
        res = df.mean(axis=1, level=0) \
            .mul(df.count(axis=1, level=0)) \
            .div(df.count(axis=1, level=0).sum(axis=1), axis=0)\
            .sum(axis=1)

        return res

    def bootstrap_wo_events(self, block_size=None):
        """Exclude event windows from `data` and bootstrap it.

        Parameters
        ----------
        block_size : int

        Returns
        -------
        EventStudy

        """
        if block_size is None:
            block_size = min(10, int(np.sqrt(len(self.data))))

        n_blocks = int(np.ceil(len(self.data) / block_size))

        boot_from = self.data\
            .where(~self.mark_event_windows(excl_ambig=False))\
            .dropna(how="all")
        rnd_choice = np.random.choice(np.arange(len(boot_from) - block_size),
                                      n_blocks * 2,
                                      replace=True)

        booted = pd.concat(
            [boot_from.iloc[p:p+block_size] for p in rnd_choice],
            axis=0, ignore_index=True
        )

        while booted.count().lt(self.data.count()).any():
            p = np.random.choice(np.arange(len(boot_from) - block_size))
            booted = booted.append(boot_from.iloc[p:p+block_size],
                                   ignore_index=True)

        n_events = self.events.count()
        events = pd.concat(
            {c: pd.Series(
                index=np.random.choice(np.arange(len(booted)), v,
                                       replace=False),
                data=1)
             for c, v in n_events.iteritems()},
            axis=1
        )

        return EventStudy(booted, events, self.window, self.event_date_index)
