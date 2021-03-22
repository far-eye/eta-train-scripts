from optparse import OptionParser
from model_gen import ModelGen
import pandas as pd
from config import *


parser = OptionParser()
parser.add_option("-r", "--route", dest="route",
                   help="Route For which model is to be generated", metavar="route")
(options, args) = parser.parse_args()
route = options.route
if route == 'NA':
    # Hack For Bhushan and JKL
    consigner_codes = ["BHUSHAN"]
    # unique route list
    cnf_obj = Config(flag='TA')
    routes_list = []
    # Consigner Code List Iteration and query formulation
    for item in consigner_codes:
        sql_query = "select distinct route, count(id) as CntRoute from toll_booth_histories where route ilike '%{cns_code}%' group by route ".format(cns_code=item)
        unique_route = pd.read_sql_query(sql_query, cnf_obj.gts_pg_connection)
        for rt in unique_route.route.values:
            routes_list.append(rt)
    for rt in routes_list:
        try:
            model_gen = ModelGen(rt).regression_model()
        except Exception as esc:
            print(str(esc))
            continue
else:
    import pdb;pdb.set_trace()
    model_gen = ModelGen(route).regression_model()
