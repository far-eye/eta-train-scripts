from utils import get_var
import psycopg2
import mysql.connector


class Config(object):
    """
    Config Class hosts all the environment based variables for
    the project
    """
    # Postgres TA database settings
    TA_PG_HOST = get_var('TA_PG_HOST', 'dipperprod-read.canbbkmz75pp.ap-south-1.rds.amazonaws.com')
    TA_PG_PORT = get_var('TA_PG_PORT', '5432')
    TA_PG_DATABASE = get_var('TA_PG_DATABASE', 'gps_development_postgres')
    TA_PG_USERNAME = get_var('TA_PG_USERNAME', 'ec2-user')
    TA_PG_PASSWORD = get_var('TA_PG_PASSWORD', 'tester')

    # Postgres TH database settings

    TH_PG_HOST = get_var('TH_PG_HOST', 'dipperprodnew-truck-histories-replica.canbbkmz75pp.ap-south-1.rds.amazonaws.com')
    TH_PG_PORT = get_var('TH_PG_PORT', '5432')
    TH_PG_DATABASE = get_var('TH_PG_DATABASE', 'gps_development_postgres')
    TH_PG_USERNAME = get_var('TH_PG_USERNAME', 'ec2-user')
    TH_PG_PASSWORD = get_var('TH_PG_PASSWORD', 'tester')

    # MYSQL  database settings

    MSQL_HOST = get_var('MSQL_HOST', '35.154.141.143')
    MSQL_PORT = get_var('MSQL_PORT', '3306')
    MSQL_DATABASE = get_var('MSQL_DATABASE', 'dipper_development')
    MSQL_USERNAME = get_var('MSQL_USERNAME', 'root')
    MSQL_PASSWORD = get_var('MSQL_PASSWORD', 'E55B25')
    POI_CONS_FETCH_URL = get_var('POI_CONS_FETCH_URL',
                                 'http://35.154.229.215:2610/poi_histories/\
                                 nearest_by_trip_multiple_poi_show?poi_id={poi_id}&trip_id={trip_id}')

    POI_CREATE_URL = get_var('POI_CREATE_URL', "https://transportation.fareye.co/api/v2/pois")

    def __init__(self, flag=None):
        self.gts_pg_connection = self._get_pg_connection(flag)
        self.mysql_connection = self._get_mysql_connection()

    def _get_pg_connection(self, flag='TA') -> object:
        """
        Get Postgres Connection Object Using Pyscopg2
        :param flag: flag is used to distinguish between TA(GTS TA App) & TH(Truck) conf
        :return: Connection object for Postgresql Pyscopg2.
        """
        if flag == 'TA':
            connection = psycopg2.connect(user=Config.TA_PG_USERNAME,
                                          password=Config.TA_PG_PASSWORD,
                                          host=Config.TA_PG_HOST,
                                          port=Config.TA_PG_PORT,
                                          database=Config.TA_PG_DATABASE)
        else:
            connection = psycopg2.connect(user=Config.TH_PG_USERNAME,
                                          password=Config.TH_PG_PASSWORD,
                                          host=Config.TH_PG_HOST,
                                          port=Config.TH_PG_PORT,
                                          database=Config.TH_PG_DATABASE)
        return connection

    def _get_mysql_connection(self) -> object:
        """
        Get Mysql Connection Object.
        :return: Mysql Connection Object
        """
        mysql_conn = mysql.connector.connect(
            host=Config.MSQL_HOST,
            database=Config.MSQL_DATABASE,
            user=Config.MSQL_USERNAME,
            passwd=Config.MSQL_PASSWORD
        )
        return mysql_conn
