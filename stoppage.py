from config import *
import math
import pandas as pd
import requests
from datetime import datetime
from haversine import haversine
import numpy as np
import json


class StoppageAdder(object):
    """
    Utility to add stoppages to the toll booth dataframe.
    """

    def __init__(self, route, route_df):
        self.stoppage_seed_route = route_df
        self.mysql_connection = Config().mysql_connection
        self.th_conn = Config().gts_pg_connection
        self.route = route
        self.lat_lng_df = []

    def get_first_odometer_histories(self, row):
        query = "SELECT * FROM truck_histories WHERE vehicle_no = '{0}' AND ist_timestamp >= '{1}' ORDER BY ist_timestamp ASC LIMIT 1;".format(
            row['vehicle_no'], row['loading_out_time'])
        odometer_histories = pd.read_sql_query(query, self.th_conn)
        try:
            odometer_km = odometer_histories.odometer_km.values[0]
        except:
            odometer_km = None
        return odometer_km

    def get_consignee_lat_lng(self, trip_val):
        for item in trip_val:
            query = "select slug, consignee_lat, consignee_long, consigner_lat, consigner_long from consigner_trips where consignee_lat is not null and consignee_long is not null and slug='{trip_id}';".format(
                trip_id=item)
            interm_df = pd.read_sql_query(query, self.mysql_connection)
            temp_dict = {'slug': interm_df.slug.values[0], 'consignee_lat': interm_df.consignee_lat.values[0],
                         'consignee_long': interm_df.consignee_long.values[0],
                         'consigner_lat': interm_df.consigner_lat.values[0],
                         'consigner_long': interm_df.consigner_long.values[0], }
            self.lat_lng_df.append(temp_dict)
        lat_lng_df = pd.DataFrame(self.lat_lng_df)
        return lat_lng_df

    # Add to toll_booth_data
    def add_lat_lng_toll_data(self, lat_lng_df, row, label, consign):
        consignee_dat = lat_lng_df[lat_lng_df.slug == row['trip_id']]
        if label == 'lat' and consign == 'consignee':
            return consignee_dat['consignee_lat'].values[0]
        elif label == 'lng' and consign == 'consignee':
            return consignee_dat['consignee_long'].values[0]
        if label == 'lat' and consign == 'consigner':
            return consignee_dat['consigner_lat'].values[0]
        elif label == 'lng' and consign == 'consigner':
            return consignee_dat['consigner_long'].values[0]

    def _create_stoppage_pois(self, lat, lng):
        payload = dict(radius=1.0, lat=lat, lng=lng, poiType='hub', poiNickName='stopagge_poi')
        headers = {
            'authority': 'transportation-test.fareye.co',
            'accept': 'application/json',
            'x-change-case': 'true',
            'authorization': 'Bearer eyJhbGciOiJub25lIn0.eyJ1c2VyX2lkIjoxOTU0LCJuYW1lIjoiQWppdCBrdW1hciIsImVtYWlsIjoiYWppdGtyQHRhdGFzdGVlbGJzbC5jby5pbiIsInBob25lX251bWJlciI6Ijc3NjM4MDc0NDMiLCJyb2xlIjoiQ29uc2lnbmVyIiwiZXhwIjoxNjI1MDUwMjQ5fQ.',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
            'content-type': 'application/json',
        }

        response = requests.request("POST", Config().POI_CREATE_URL, headers=headers, data=payload)

        if response.status_code == 200:
            return response.json()['id']

    def get_stoppage_cons(self, stoppage_analysis_route_df, stoppage_pois):
        stoppage_df = []
        for index, row in stoppage_analysis_route_df.iterrows():
            for pois in stoppage_pois:
                url = Config().POI_CONS_FETCH_URL.format(poi_id=pois, trip_id=row['trip_id'])
                req = requests.get(url)
                data = req.json()
                if data:
                    for cons in data:
                        temp = {}
                        temp['poi_id'] = cons['poi_id']
                        temp['vehicle_no'] = cons['vehicle_no']
                        temp['id'] = 'NA'
                        temp['trip_id'] = cons['trip_id']
                        temp['odometer_km'] = cons['entry_odometer_km']
                        temp['route'] = self.route
                        temp['loading_out_time'] = row['loading_out_time']
                        temp['unloading_in_time'] = row['unloading_in_time']
                        temp['start_date'] = 'NA'
                        ist_timestamp = datetime.strptime(cons['enter_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
                        ist_timestamp = ist_timestamp.strftime(format='%Y-%m-%d %H:%M:%S')
                        temp['ist_timestamp'] = ist_timestamp
                        try:
                            temp['distance_travelled'] = cons['entry_odometer_km'] - row['odometer_km']
                        except:
                            temp['distance_travelled'] = haversine((row['consigner_lat'], row['consigner_long']),
                                                                   (cons['entry_lat'], cons['entry_long']))

                        temp['createdAt'] = np.nan
                        temp['updatedAt'] = np.nan
                        stoppage_df.append(temp)
        stoppage_df = pd.DataFrame(stoppage_df)
        stoppage_df['ist_timestamp'] = pd.to_datetime(stoppage_df['ist_timestamp'], format='%Y-%m-%d %H:%M:%S')
        stoppage_df['time_taken'] = (stoppage_df['ist_timestamp'] - stoppage_df[
            'loading_out_time']).dt.total_seconds() / 86400
        return stoppage_df

    def generate_stoppage_poi(self):
        """
        Generate Stoppage Poi for
        :return:
        """
        stoppage_analysis_route_df = self.stoppage_seed_route
        query = "SELECT id, stoppage_identifier, latitude, longitude, no_of_vehicles,  SQRT(" + \
                "POW(69.1 * (latitude - (select consigner_lat from consigner_trips WHERE  origin='{origin}' AND " \
                "destination='{destniation}' AND consigner_code='{consigner_code}' AND loading_out_time is not null limit 1)), " \
                "2) +" + \
                "POW(69.1 * ((select consigner_long from consigner_trips WHERE  origin='{origin}' AND destination='{" \
                "destniation}' AND consigner_code='{consigner_code}' AND loading_out_time is not null limit 1) - longitude) * " \
                "COS(latitude / 57.3), 2)) AS distance" + \
                "FROM stoppage_points  where stoppage_identifier='{stoppage_route}' and no_of_vehicles > 10  order by " \
                "no_of_vehicles desc limit {limit}; "

        total_count = self.stoppage_seed_route.shape[0]
        limit_vehicle = total_count
        if total_count >= 100:
            limit_vehicle = math.ceil(total_count / 3)

        route_split = self.route.split('#')
        # Origin, Destination, Consigner Code
        origin = route_split[0]
        destination = route_split[1]
        consigner_code = route_split[2]

        stoppage_route = origin + '-' + destination
        stoppage_route = stoppage_route.upper()

        query = query.format(origin=origin, destination=destination,
                             consigner_code=consigner_code, stoppage_route=stoppage_route, limit=limit_vehicle)

        stoppage_df = pd.read_csv(query, self.mysql_connection)
        stoppage_pois = list()
        for index, row in stoppage_df.iterrows():
            lat = row['latitude']
            lng = row['longitude']

            id = self._create_stoppage_pois(lat, lng)

            if id:
                stoppage_pois.append(id)

        with open('stoppage_poi/' + stoppage_route + '.json', 'wb') as write:
            json.dump(stoppage_pois, write)

        stoppage_analysis_route_df['odometer_km'] = stoppage_analysis_route_df.apply(self.get_first_odometer_histories,
                                                                                     axis=1)
        trip_id_vals = set(stoppage_analysis_route_df['trip_id'].values)

        lat_lng_df = self.get_consignee_lat_lng(trip_val=trip_id_vals)

        stoppage_analysis_route_df['consignee_lat'] = stoppage_analysis_route_df.apply(self.add_lat_lng_toll_data,
                                                                                       args=(
                                                                                       lat_lng_df, 'lat', 'consignee',),
                                                                                       axis=1)
        stoppage_analysis_route_df['consignee_long'] = stoppage_analysis_route_df.apply(self.add_lat_lng_toll_data,
                                                                                        args=(lat_lng_df, 'lng',
                                                                                              'consignee',),
                                                                                        axis=1)
        stoppage_analysis_route_df['consigner_lat'] = stoppage_analysis_route_df.apply(self.add_lat_lng_toll_data,
                                                                                       args=(
                                                                                       lat_lng_df, 'lat', 'consigner',),
                                                                                       axis=1)
        stoppage_analysis_route_df['consigner_long'] = stoppage_analysis_route_df.apply(self.add_lat_lng_toll_data,
                                                                                        args=(lat_lng_df, 'lng',
                                                                                              'consigner',),
                                                                                        axis=1)
        stoppage_df = self.get_stoppage_cons(stoppage_analysis_route_df, stoppage_pois)
        return stoppage_df

