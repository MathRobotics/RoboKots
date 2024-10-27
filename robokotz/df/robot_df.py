#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.07.20 Created by T.Ishigaki

import numpy as np
import polars as pl

class RobotDF:
  def __init__(self, names_, aliases_, separator_ = "_"):
    self.names = names_
    self.aliases = aliases_
    self.separator = separator_

    self.df = pl.DataFrame()
    self.set_df()
    
  def add_row(self, data):
    new_row = pl.DataFrame([data])
    self.df = self.df.vstack(new_row)
    
  def set_df(self):
    for name in self.names:
      for a in self.aliases:
        alias_name = name + self.separator + a
        self.df = self.df.with_columns([pl.Series(name=alias_name, dtype=pl.List(pl.Float64))])
