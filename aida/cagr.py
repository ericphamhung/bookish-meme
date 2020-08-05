import pandas as pd
import numpy as np

def cagr(del_t, v1, v2):
    if v2 == 0:
        return -np.inf
    #print((v2/v1)**(365.25/(del_t))-1.0)
    return (v2/v1)**(365.25/(del_t))-1.0

def CAGR(times, vols_or_sales):
    assert len(times) == len(vols_or_sales)
    cagrs = []
    for i in range(1, len(times)):
        tt = (times[i]-times[i-1]).days
        s1, s2 = vols_or_sales[i-1], vols_or_sales[i]
        cc = cagr(tt, s1, s2)
        cagrs.append(cc)
    return cagrs 