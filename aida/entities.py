import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import pyro

__allowed_types__ = ['time', 'client', 'rep', 'store', 'product-category']
__allowed_times__ = ['all', 'year', 'year-to-date', 'month', 'week', 'day']


agg_hier = {'product':['sku', 'category', '??'],
            'time1':['invoice', 'day', 'week', 'month', 'year', 'all'],
            'time2':['invoice', 'day', 'month', 'year', 'all'],
            'time3':['invoice', 'day', 'week', 'year', 'all'], 
            'time4':['invoice', 'day', 'year', 'all'],
            'ytd1':['invoice', 'day', 'week', 'month', 'ytd'],
            'ytd2':['invoice', 'day', 'month', 'ytd'],
            'ytd3':['invoice', 'day', 'week', 'ytd'],
            'ytd4':['invoice', 'day', 'ytd'],
            'customer':['cust', 'cust_agg', 'all'],
            'rep':['rep', 'all'],
            }

class SalesFrame:

    def __init__(self, 
                filen, agg_inv = False):
        
        self.df = pd.read_csv(filen)
        self.columns = self.df.columns.tolist()
        self.units = [i if 'unit' in c else None for i, c in self.columns]
        self.sales = [i if 'sale' in c else None for i, c in self.columns]
        self.price = [i if 'price' in c else None for i, c in self.columns]
        self.agg_cols = [c for c in self.columns if 'date' in c]
        self.agg_cols += [c for c in self.columns if 'time' in c]
        self.agg_cols += [c for c in self.columns if 'prod' in c]
        self.agg_cols += [c for c in self.columns if 'rep' in c]
        self.agg_cols += [c for c in self.columns if 'stor' in c]
        if agg_inv:
            self.agg_cols += [c for c in self.columns if 'inv' in c]
            assert len([c for c in self.columns if 'inv' in c])>0
        self.agg_cols += [c for c in self.columns if 'prod' in c]
        self.agg_cols += [c for c in self.columns if 'cat' in c]
        assert len(self.units)<=1
        assert len(self.price)<=1
        assert len(self.sales)<=1
        if (len(self.units)>0)*1+(len(self.sales)>0)*1+(len(self.price)>0)*1<2:
            raise ValueError('There do not seem to be 2 of three of units, sales, and price')
        if len(self.units) == 0:
            self.columns.append('units')
            self.units = [len(self.columns)-1]
            self.df['units'] = self.df[self.columns[self.sales]]/self.df[self.columns[self.price]]
        elif len(self.price) == 0:
            self.columns.append('price')
            self.price = [len(self.columns)-1]
            self.df['price'] = self.df[self.columns[self.sales]]/self.df[self.columns[self.units]]
        elif len(self.sales) == 0:
            self.columns.append('sales')
            self.sales = [len(self.columns)-1]
            self.df['sales'] = self.df[self.columns[self.units]]*self.df[self.columns[self.price]]
        
    def price_makes_sense(self, over):
        pass

    def aggregrate(self, over):
        if self.price_makes_sense(over):
            pass 
    

