import pandas as pd
from datetime import datetime
import datetime as dt
import math
from stoppage import StoppageAdder
import numpy as np
from scipy import stats
from config import Config


class DSGenerator(object):
    """
    This Class Serves as utility class to generate dataset which is used to
    run and train model.
    """

    POI_CUTOFF_PERCENTAGE = 0.70
    Z_SCORE_THRESHOLD = 3
    ONE_HOT_ENCODED_POI_COLS = None

    def __init__(self, route):
        self.ta_connection = Config(flag='TA').gts_pg_connection
        self.route = route
        self.toll_data = self.process_routes_get_toll_booths()
        self.route_df = self.toll_data[['vehicle_no', 'route', 'trip_id', 'loading_out_time', 'unloading_in_time']].drop_duplicates()
        self.stoppage_adder_obj = StoppageAdder(self.route, self.route_df)
        self.vars_data = self._generate_vars(toll_data=self.toll_data)

    @staticmethod
    def drop_features_na(dataframe, drop_cols=None):
        """
        This function drops features set which have more than 80% of missing data and also drops columns
        which are mentioned.
        """
        drop_columns = []
        if drop_cols:
            for col in drop_cols:
                drop_columns.append(col)
        for item in dataframe.columns:
            perc_missing_data = dataframe[item].isna().sum() / dataframe.shape[0] * 100
            if perc_missing_data >= 80.0:
                print("========= Column :- {col} and Percentage:- {perc} is to be dropped".format(col=item,
                                                                                                  perc=perc_missing_data))
                drop_columns.append(item)
        dataframe.drop(drop_columns, axis=1, inplace=True)
        return dataframe

    def process_routes_get_toll_booths(self):
        sql_query = "select * from toll_booth_histories where route='{route}';".format(route=self.route)
        toll_booth_data = pd.read_sql_query(sql_query, self.ta_connection)
        return toll_booth_data

    def _drop_null_datetime(self, toll_data):
        toll_data = toll_data.drop(toll_data[toll_data.ist_timestamp == np.nan].index)
        toll_data = toll_data.drop(toll_data[toll_data.loading_out_time == np.nan].index)
        toll_data.drop(toll_data[toll_data['ist_timestamp'] > toll_data['unloading_in_time']].index,
                       inplace=True)
        return toll_data

    def poi_based_filtering(self, toll_data):
        """
        POI based filtering is required to remove outliers such that
        we have poi id which contains the data rather than the ones which
        have very few records.
        """
        poi_df = pd.DataFrame(toll_data.poi_id.value_counts(), columns=["poi_id", "count"]).reset_index()
        total_trip_id = toll_data.trip_id.nunique()

        poi_cutoff = math.ceil(total_trip_id * DSGenerator.POI_CUTOFF_PERCENTAGE)

        poi_df = poi_df[poi_df.poi_id >= poi_cutoff]

        toll_data = toll_data[toll_data.poi_id.isin(poi_df['index'])]
        return toll_data

    @staticmethod
    def z_score_based_imputation(toll_data):
        for poi in toll_data.poi_id.unique():
            filtered_poi = toll_data[toll_data.poi_id == poi]
            z = np.abs(stats.zscore(filtered_poi.distance_travelled))
            index = filtered_poi[(z > DSGenerator.Z_SCORE_THRESHOLD)].index
            print("Z Score {z_score} for poi id:- {poi_id}".format(z_score=index, poi_id=poi))
            toll_data.loc[
                toll_data.index.isin(index), 'distance_travelled'] = filtered_poi.distance_travelled.median()
        return toll_data

    def _generate_vars(self, toll_data):
        # Adding Stoppage Data
        stoppage_df = self.stoppage_adder_obj.generate_stoppage_poi()
        filtered_stoppage_df = stoppage_df[stoppage_df.time_taken > 0]
        filtered_stoppage_df.reset_index(inplace=True, drop=True)
        toll_data = toll_data.append(filtered_stoppage_df, ignore_index=True)
        # Get loading Out Time of Day ( Night->0, Morning->1, Afternoon->2, Evening->3)
        toll_data = toll_data.assign(loading_out_time_of_day=pd.cut(toll_data.loading_out_time.dt.hour,
                                                                    [0, 6, 12, 18, 23],
                                                                    labels=[0, 1, 2, 3],
                                                                    include_lowest=True))

        # Get Toll booth entry In Time of Day ( Night->0, Morning->1, Afternoon->2, Evening->3)

        toll_data = toll_data.assign(is_time_day=pd.cut(toll_data.ist_timestamp.dt.hour,
                                                        [0, 6, 12, 18, 23],
                                                        labels=[0, 1, 2, 3],
                                                        include_lowest=True))

        # Get Toll booth entry In day of month [1...31]

        toll_data['entry_day_week'] = toll_data.ist_timestamp.dt.day

        # Get Toll booth entry In day of month [1...31]

        toll_data['loading_out_day_week'] = toll_data.loading_out_time.dt.day

        toll_data['ist_weekday'] = toll_data['ist_timestamp'].dt.weekday
        toll_data['ist_week_of_year'] = toll_data['ist_timestamp'].dt.weekofyear
        toll_data['ist_hour'] = toll_data['ist_timestamp'].dt.hour
        toll_data['ist_week_hour'] = toll_data['ist_weekday'] * 24 + toll_data['ist_hour']

        toll_data['loading_out_weekday'] = toll_data['loading_out_time'].dt.weekday
        toll_data['loading_out_wek_of_year'] = toll_data['loading_out_time'].dt.weekofyear
        toll_data['loading_out_hour'] = toll_data['loading_out_time'].dt.hour
        toll_data['loading_out_week_hour'] = toll_data['loading_out_weekday'] * 24 + toll_data['loading_out_hour']
        toll_data['time_from_loading_point'] = (
                toll_data['ist_timestamp'] - toll_data['loading_out_time']).dt.total_seconds()
        toll_data['time_from_loading_point'] = toll_data['time_from_loading_point'] / 3600
        toll_data['avg_speed'] = toll_data['distance_travelled'] / toll_data['time_from_loading_point']
        toll_data['time_taken_actual'] = (toll_data['unloading_in_time'] - toll_data[
            'ist_timestamp']).dt.total_seconds() / 3600
        toll_data['time_taken'] = toll_data['time_taken'] * 24
        toll_data.drop_duplicates(subset=["vehicle_no", "poi_id"], inplace=True)

        toll_data = self._drop_null_datetime(toll_data)
        toll_data = self.poi_based_filtering(toll_data)
        toll_data = DSGenerator.z_score_based_imputation(toll_data)

        toll_data = DSGenerator.drop_features_na(toll_data)
        one_hot_encoded_poi_id = pd.get_dummies(toll_data['poi_id'], prefix='poi')
        for item in one_hot_encoded_poi_id.columns:
            toll_data[item] = one_hot_encoded_poi_id[item]

        DSGenerator.ONE_HOT_ENCODED_POI_COLS = list(one_hot_encoded_poi_id.columns)

        return toll_data
