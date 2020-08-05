from dataclasses import dataclass, field
import pandas as pd
from typing import List
import numpy as np

curr_date = pd.to_datetime('2019-08-31')

@dataclass
class Item:
    sku: int
    units: float
    sale: float
    price: float = field(init=False)

    def __post_init__(self):
        self.price = self.sale/self.units

@dataclass
class SKU_summary:
    sku: int
    first_sale: pd.datetime
    last_sale: pd.datetime
    is_active: bool = field(init=False)
    sales_items: List[Item]
    total_sales: float = field(init=False)
    total_units: float = field(init=False)
    avg_price: float = field(init=False)

    def __post_init__(self):
        self.total_sales = 0.0
        self.total_units = 0.0
        for item in self.sales_items:
            assert item.sku == self.sku
            self.total_sales += item.sale
            self.total_units += item.units
        self.avg_price = self.total_sales/self.total_units

        

@dataclass
class Container:
    skus: List[int]
    units: List[float]
    sales: List[float]


@dataclass
class Invoice:
    items = List[Item]
    sale_amount: float = field(init=False)

    def __post_init__(self):
        self.sale_amount = 0.0
        for item in self.items:
            self.sale_amount += item.sale