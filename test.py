#%matplotlib inline
import matplotlib.pyplot as plt
import quandl
import datetime
from datetime import date
import pandas as pd
import numpy as np
import csv
from sharpeOpt import Optimizer


stocksymlist = ["MSFT", "AXP", "BA", "CAT", "CVX", "CSCO", "KO", "DIS",  "XOM", "GE", "GS",
                "HD", "IBM", "JNJ", "JPM", "MCD", "MRK", "NKE", "PFE", "PG", "UTX", "UNH", "VZ",
                "V", "WMT"]

opt = Optimizer(stocksymlist)
w, s = opt.ascent(1000, 0.0001)

