import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import get_data, pitcher_class

p_throws_map = {'R': 0, 'L': 1}
pt_data = get_data.get_pitcher_data()
p_dict = pitcher_class.format_pitcher_data(pt_data)
new_ds = pitcher_class.process_batch_avg(p_dict)

rv_data = get_data.get_runvalue_data()

full_ds = get_data.join_ds(new_ds, rv_data) #join data

# Apply one-hot encoding to pitches, apply binary to L/R
one_hot = pd.get_dummies(full_ds['pitch_name'])
full_ds = full_ds.drop('pitch_name',axis = 1)
full_ds = full_ds.join(one_hot)
full_ds['p_throws'] = [p_throws_map[x] for x in full_ds['p_throws']]

#rearrange cols
full_ds = full_ds['player_name','p_throws','Splitter', '4-Seamer', 'Curveball', 'Cutter', 'Sinker', 'Changeup', 
                  'Slider', 'Sweeper', 'Slurve', 'Forkball', 'Screwball', 'Knuckleball','release_speed','release_pos_x',
                  'release_pos_z','pfx_x','pfx_z','release_extension','spin_axis','useage%','pitches','whiff_percent','est_woba',
                  'run_value_per_100']

lg_avs = get_data.league_avgs(full_ds)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear()