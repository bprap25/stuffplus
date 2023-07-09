from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import pitching_stats
from pybaseball import cache
from datetime import date
from pitcher_class import Pitcher
import pandas as pd

# rv_to_pitch_map = {'4-Seam Fastball': '4-Seamer', 'Cutter':'Cutter','Sinker': 'Sinker'}

cache.enable()
raw_data = statcast(start_dt='2023-03-30', end_dt= str(date.today()))
pitch_data = raw_data[['player_name','pitch_type','release_speed','release_pos_x','release_pos_z','spin_dir','p_throws','pfx_x',
                       'pfx_z','release_spin_rate','release_extension','pitch_name','spin_axis']]
pitch_data['pfx_x'] = pitch_data['pfx_x'].apply(lambda x: x* 12)
pitch_data['pfx_z'] = pitch_data['pfx_z'].apply(lambda z: z* 12)

# pitch_data.drop(pitch_data[(pitch_data['pitch_name'] == None) | (pitch_data['pitch_name'] == 'Other') | (pitch_data['pitch_name'] == 'Eephus')
#                            | (pitch_data['pitch_name'] == 'Pitch Out')], inplace=True)

raw_run_val = pd.read_csv('pitch-arsenal-stats.csv')
print(raw_run_val.columns)

run_val = raw_run_val[['last_name','first_name','pitch_name','run_value_per_100']]

raw_run_val.to_csv('test.csv')

print(run_val['pitch_name'].unique())

print(pitch_data['pitch_name'].unique())