from get_dataset import DSGenerator
from sklearn.preprocessing import StandardScaler
from config import Config
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
import pickle


class ModelGen(object):
    PCA_COMPONENTS = 3
    TRAIN_COLUMNS = ['time_taken', 'distance_travelled', 'loading_out_time_of_day', 'is_time_day',
                     'entry_day_week', 'loading_out_day_week', 'ist_weekday', 'ist_week_of_year',
                     'ist_hour', 'ist_week_hour', 'loading_out_wek_of_year',
                     'loading_out_hour', 'loading_out_week_hour', 'avg_speed', 'time_from_loading_point'
                     ]
    TEST_COLUMNS = 'time_taken_actual'
    ONE_HOT_ENCODE_CLUSTER = None

    def __init__(self, route):
        self.ds_gen_obj = DSGenerator(route)
        self.model_data = self.ds_gen_obj.vars_data
        self.scaler = StandardScaler()
        self.one_hot_encoded_poi_cols = self.ds_gen_obj.ONE_HOT_ENCODED_POI_COLS
        self.th_connection = Config(flag='TH').th_pg_connection
        self.model_routes = route.replace('#', '')

    def generate_train_columns(self):
        for item in ModelGen.ONE_HOT_ENCODE_CLUSTER:
            ModelGen.TRAIN_COLUMNS.append(item)

        for item in self.one_hot_encoded_poi_cols:
            ModelGen.TRAIN_COLUMNS.append(item)

    def _generate_cluster_model(self):
        scaled_data = self.scaler.fit_transform(self.model_data[['time_taken',
                                                                 'distance_travelled', 'loading_out_time_of_day',
                                                                 'is_time_day', 'entry_day_week', 'loading_out_day_week',
                                                                 'ist_weekday', 'ist_week_of_year', 'ist_hour',
                                                                 'ist_week_hour', 'loading_out_wek_of_year',
                                                                 'loading_out_hour', 'loading_out_week_hour',
                                                                 'avg_speed', 'time_from_loading_point']])
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(scaled_data)
        diff = (self.model_data.distance_travelled.max() - self.model_data.distance_travelled.min()) / \
               self.model_data.distance_travelled.median()
        db_scan = DBSCAN(eps=diff).fit(principalComponents)
        cluster_model = 'models/cluster_model_{route}.sav'.format(route=self.model_routes)
        pickle.dump(db_scan, open(cluster_model, 'wb'))
        self.model_data['cluster'] = db_scan.labels_
        return self.model_data

    def generate_one_hot_encoded_cluster(self):
        """
        Generate One Hot Encoded Cluster Columns for a df.
        :return: Pandas DataFrame
        """
        one_hot_encoded_cluster = pd.get_dummies(self.model_data['cluster'], prefix='cluster')
        for item in one_hot_encoded_cluster.columns:
            self.model_data[item] = one_hot_encoded_cluster[item]

        ModelGen.ONE_HOT_ENCODE_CLUSTER = list(one_hot_encoded_cluster.columns)
        return self.model_data

    def vehicle_based_train_test_split(self, test_data_perc):
        X = self.model_data[ModelGen.TRAIN_COLUMNS]
        Y = self.model_data[ModelGen.TEST_COLUMNS]

        if not test_data_perc:
            test_data_perc = 0.3

        train_data_perc = 1 - test_data_perc
        vehicle_unique = self.model_data.vehicle_no.unique()
        split_records = math.ceil(self.model_data.vehicle_no.nunique() * train_data_perc)

        train_vehicle = vehicle_unique[0:split_records]
        test_vehicle = vehicle_unique[split_records:]

        train_index = self.model_data[self.model_data.vehicle_no.isin(train_vehicle)].index
        test_index = self.model_data[self.model_data.vehicle_no.isin(test_vehicle)].index

        train_x = X[X.index.isin(train_index)]
        test_x = X[X.index.isin(test_index)]
        y_train = Y[Y.index.isin(train_index)]
        y_test = Y[Y.index.isin(test_index)]

        return train_x, test_x, y_train, y_test

    def drop_correlated_features(self):
        self.model_data = self._generate_cluster_model()
        self.model_data = self.generate_one_hot_encoded_cluster()
        # Generate test_cols
        self.generate_train_columns()
        X_train, X_test, y_train, y_test = self.vehicle_based_train_test_split(test_data_perc=0.2)

        # creating set to hold the correlated features
        corr_features = set()

        # create the correlation matrix (default to pearson)
        corr_matrix = X_train.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= 0.90:
                    colname = corr_matrix.columns[i]
                    if not colname.startswith('cluster'):
                        corr_features.add(colname)

        X_train.drop(labels=corr_features, axis=1, inplace=True)
        X_test.drop(labels=corr_features, axis=1, inplace=True)

        return X_train, X_test, y_train, y_test

    # Function To Convert seconds to days
    def convert_prediction_to_days(self, row, label):
        actual = pd.Timedelta(seconds=row[label])
        return actual

    def get_metrics(self, y_test, y_pred):
        metrics_dict = dict(rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                            mse=metrics.mean_squared_error(y_test, y_pred),
                            mae=metrics.mean_absolute_error(y_test, y_pred))
        return metrics_dict

    def get_rmse(self, df):
        rmse_poi_wise = []
        for poi in df.poi_id.unique():
            filtered_poi = df[df.poi_id == poi]
            rmse_poi_wise.append({'poi_id': poi, 'rmse': np.sqrt(
                metrics.mean_squared_error(filtered_poi.time_taken_actual, filtered_poi.time_taken_predicted)),
                                  'median_distance': filtered_poi.distance_travelled.median()})
        return rmse_poi_wise

    def regression_model(self):
        X_train, X_test, y_train, y_test = self.drop_correlated_features()
        X_scaled_train = self.scaler.fit_transform(X_train)
        X_scaled_test = self.scaler.fit_transform(X_test)
        # Performing Grid Search CV to get best parameters
        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_scaled_train, y_train)
        best_params = grid_result.best_params_
        best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                       coef0=0.1, shrinking=True,
                       tol=0.001, cache_size=200, verbose=False, max_iter=-1)
        regressor = best_svr.fit(X_scaled_train, y_train)
        y_pred = regressor.predict(X_scaled_test)

        # Regression Model Pickling.
        regression_model = 'models/regressor_{route}.sav'.format(route=self.model_routes)
        pickle.dump(regressor, open(regression_model, 'wb'))

        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

        metrics = self.get_metrics(y_test, y_pred)
        print('RMSE is : ', metrics['rmse'])
        print('MAE is:', metrics['mae'])
        predicted_data = self.model_data[self.model_data.index.isin(df.index)]
        predicted_data['time_taken_predicted'] = df['Predicted']
