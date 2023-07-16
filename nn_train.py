import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import get_data, pitcher_class
from torch.utils.data import DataLoader, TensorDataset

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
full_ds = full_ds[['player_name','p_throws','Splitter', '4-Seamer', 'Curveball', 'Cutter', 'Sinker', 'Changeup', 
                  'Slider', 'Sweeper', 'Slurve', 'Forkball', 'Screwball', 'Knuckleball','release_speed','release_pos_x',
                  'release_pos_z','pfx_x','pfx_z','release_extension','spin_axis','useage %','pitches','whiff_percent','est_woba',
                  'run_value_per_100']]
lg_avs = get_data.league_avgs(full_ds)

new_plus = get_data.generate_plus(full_ds,lg_avs)
full_ds['calc_plus'] = new_plus
full_ds.drop(columns=['whiff_percent','est_woba','run_value_per_100'])
full_ds = full_ds[full_ds['pitches'] >= 20]

init_inp = full_ds[['p_throws','Splitter', '4-Seamer', 'Curveball', 'Cutter', 'Sinker', 'Changeup', 
                  'Slider', 'Sweeper', 'Slurve', 'Forkball', 'Screwball', 'Knuckleball','release_speed','release_pos_x',
                  'release_pos_z','pfx_x','pfx_z','release_extension','spin_axis']]
X = torch.tensor(init_inp.to_numpy())
X = X.type(torch.float32)
init_out = full_ds[['calc_plus']]
Y = torch.tensor(init_out.to_numpy())
Y = Y.type(torch.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=.66, random_state=42)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.hidden1 = nn.Linear(20,8)
        self.output = nn.Linear(8,1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.output(x)
        return x

model = model()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=1000)
test_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))


for epoch in range(200):
    for X,Y in train_loader:
        pred = model(X)
        loss = mse_loss(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pt')