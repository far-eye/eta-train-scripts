from optparse import OptionParser
from model_gen import ModelGen


parser = OptionParser()
parser.add_option("-r", "--route", dest="route",
                  help="Route For which model is to be generated", metavar="route")
(options, args) = parser.parse_args()
model_gen = ModelGen(options.route).regression_model()