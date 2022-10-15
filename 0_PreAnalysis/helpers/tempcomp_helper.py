from enum import unique
import os
import pathlib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import ndimage
import pickle
import matplotlib.pyplot as plt

def get_comp_factors (test_df, sensor_id_list, unique_rtds):
  SG_comp_mapping = {1:1, 4:1, 5:1, 2:6, 6:6} #SG_id : RTD_id for 2RTD case.
  comp_factors = dict()

  for sensor_id in sensor_id_list:
    if not unique_rtds: #If we only have RTD 1 and RTD 6 available
      rtd_col = "RTD " + str(SG_comp_mapping[sensor_id]) + " (V) (normalized)"
    else:
      rtd_col = "RTD " + str(sensor_id) + " (V) (normalized)"
    sg_col = "SG " + str(sensor_id) + " (V) (normalized)"
    rtd_dat = test_df[rtd_col]
    sg_dat = test_df[sg_col]

    if sg_col + " (compensated)" in test_df:
      _ = input("Compensated data already exists in the dataframe. Please cancel and review the .csv file")
    if rtd_col == "RTD 6 (V) (normalized)":
      print ("Uniform filter applied to RTD 6.")
      rtd_dat = ndimage.uniform_filter1d(rtd_dat, size = 100)

    N = rtd_dat.size
    fun = lambda x: 1/N * np.sum(np.abs(sg_dat - x[0] - x[1]*rtd_dat - x[2]*rtd_dat**2 - x[3]*rtd_dat**3))
    res = minimize(fun, (1, 1, 1, 1))
    
    comp_factors[sensor_id] = [res.x[0], res.x[1], res.x[2], res.x[3]]
  return comp_factors


def add_compensated_data_to_df (test_df, comp_factors, unique_rtds):
  SG_comp_mapping = {1:1, 4:1, 5:1, 2:6, 6:6} #SG_id : RTD_id for 2RTD case.

  for sensor_id, coefficients in comp_factors.items():
    if not unique_rtds: #If we only have RTD 1 and RTD 6 available
      rtd_col = "RTD " + str(SG_comp_mapping[sensor_id]) + " (V) (normalized)"
    else:
      rtd_col = "RTD " + str(sensor_id) + " (V) (normalized)"
    sg_col = "SG " + str(sensor_id) + " (V) (normalized)"
    rtd_dat = test_df[rtd_col]
    sg_dat = test_df[sg_col]

    if rtd_col == "RTD 6 (V) (normalized)":
      print ("Uniform filter applied to RTD 6.")
      rtd_dat = ndimage.uniform_filter1d(rtd_dat, size = 100)

    comp_sg_dat = sg_dat - coefficients[0] - coefficients[1]*rtd_dat - coefficients[2]*rtd_dat**2 - coefficients[3]*rtd_dat**3
    col_num = test_df.columns.get_loc(sg_col)
    test_df.insert(col_num+1, sg_col+" (compensated)", comp_sg_dat)

  return test_df


def add_compensated_data_to_df_infrequent (test_df, comp_factors):
  for sensor_id, coefficients in comp_factors.items():
    assert "RTD 2 (V) (normalized)" in test_df.columns #This function is only available when data is collected from all RTDs
    
    rtd_col = "RTD " + str(sensor_id) + " (V) (normalized)"
    sg_col = "SG " + str(sensor_id) + " (V) (normalized)"
    
    if rtd_col == "RTD 6 (V) (normalized)":
      print ("Uniform filter applied to RTD 6.")
      test_df[rtd_col] = ndimage.uniform_filter1d(test_df[rtd_col], size = 100)

    infrequent_rtd_dat = test_df.loc[::100, rtd_col]
    reindexed = infrequent_rtd_dat.reindex(range(test_df.shape[0]), fill_value=None)
    filled = reindexed.ffill()

    rtd_dat = filled
    sg_dat = test_df[sg_col]

    comp_sg_dat = sg_dat - coefficients[0] - coefficients[1]*rtd_dat - coefficients[2]*rtd_dat**2 - coefficients[3]*rtd_dat**3
    col_num = test_df.columns.get_loc(sg_col)
    test_df.insert(col_num+1, sg_col+" (compensated)", comp_sg_dat)

  return test_df