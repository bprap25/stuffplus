from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import pitching_stats
from pybaseball import cache
from datetime import date
import pandas as pd



#global map to call for 1:1 mapping between pitch and rv datasets
pitch_to_rv_map = {'Split-Finger': 'Splitter', '4-Seam Fastball': '4-Seamer', 'Slow Curve': 'Curveball', 'Knuckle Curve': 'Curveball',
                   'Cutter': 'Cutter', 'Sinker': 'Sinker', 'Changeup': 'Changeup', 'Slider': 'Slider', 'Curveball': 'Curveball',
                   'Sweeper': 'Sweeper', 'Slurve': 'Slurve', 'Forkball': 'Forkball', 'Screwball': 'Screwball', 'Knuckleball': 'Knuckleball'}
cache.enable()

def get_pitcher_data():
    """
    Args:None
    Returns: pitcher dataframe of all pitches thrown from the start of 2023 season to current data
    """
    raw_data = statcast(start_dt='2023-03-30', end_dt= str(date.today()))
    #select columns, remove overhead data
    p_data = raw_data[['player_name','pitch_name','release_speed','release_pos_x','release_pos_z','p_throws','pfx_x',
                       'pfx_z','release_spin_rate','release_extension','spin_axis']]

    #filter out position player pitches
    p_data = p_data[ (p_data['pitch_name'] == '4-Seam Fastball') | (p_data['pitch_name'] == 'Changeup') | (p_data['pitch_name'] == 'Curveball') 
                    | (p_data['pitch_name'] == 'Cutter') | (p_data['pitch_name'] == 'Forkball')
                    | (p_data['pitch_name'] == 'Knuckle Curve') | (p_data['pitch_name'] == 'Knuckleball')| (p_data['pitch_name'] == 'Screwball')
                    | (p_data['pitch_name'] == 'Sinker')| (p_data['pitch_name'] == 'Slider')| (p_data['pitch_name'] == 'Slow Curve')
                    | (p_data['pitch_name'] == 'Slurve')| (p_data['pitch_name'] == 'Split-Finger')| (p_data['pitch_name'] == 'Sweeper')]

    p_data['pfx_x'] = p_data['pfx_x'].apply(lambda x: x* 12) #vertical/horizontal break numbers are typically understood in inches, not ft
    p_data['pfx_z'] = p_data['pfx_z'].apply(lambda z: z* 12)
    p_data['pitch_name'] = p_data['pitch_name'].apply(lambda y: pitch_to_rv_map[y]) #make pitch types 1:1 mapping between rv and pitch data

    return p_data.dropna(how = 'any')

def get_runvalue_data():
    """
    Args: None
    Returns: Run Value dataframe of each pitch in MLB for the specified year
    """
    raw_run_val = pd.read_csv('data/pitch-arsenal-stats.csv') #access to csv restricted, must do manual pull
    run_val = raw_run_val[['last_name','first_name','pitch_name','pitches','whiff_percent','est_woba','run_value_per_100']]

    run_val = run_val[(run_val['pitch_name'] == '4-Seamer') | (run_val['pitch_name'] == 'Cutter')| (run_val['pitch_name'] == 'Sinker')
                  | (run_val['pitch_name'] == 'Changeup')| (run_val['pitch_name'] == 'Slider')| (run_val['pitch_name'] == 'Curveball')
                  | (run_val['pitch_name'] == 'Splitter')| (run_val['pitch_name'] == 'Sweeper')| (run_val['pitch_name'] == 'Slurve')
                  | (run_val['pitch_name'] == 'Forkball')| (run_val['pitch_name'] == 'Screwball')| (run_val['pitch_name'] == 'Knuckleball')]

    run_val['player_name'] = run_val['last_name'] + ',' + run_val['first_name']
    run_val.drop(['last_name','first_name'], axis = 1, inplace=True) #format dataset to have player_name in same format as p_data

    run_val['run_value_per_100'] = run_val['run_value_per_100'].apply(lambda x: x* -1)
    return run_val.dropna(how = 'any')

def join_ds(inp, out):
    new_df = pd.merge(inp, out,  how='outer', on = ['player_name','pitch_name'])
    new_df.dropna(how = 'any', inplace= True)
    return new_df

def league_avgs(rv_ds):
    pitches = 0
    tot_whiffs = 0
    tot_woba = 0
    tot_rv = 0
    for index, row in rv_ds.iterrows():
        whiff_dec = row['whiff_percent']/100
        tot_whiffs += int(row['pitches'] * whiff_dec)
        pitches += row['pitches']
        tot_rv += row['run_value_per_100']
        tot_woba += int(row['pitches'] * row['est_woba'])
    avg_whiff = (tot_whiffs/pitches) * 100
    avg_woba = tot_woba/pitches
    avg_rv_100 = tot_rv/len(rv_ds)

    return [avg_whiff, avg_woba, avg_rv_100]

def generate_plus(ds,avs):
    av_whiff = avs[0]
    av_woba = avs[1]
    av_rv_100 = avs[2]
    ls = []
    for index,row in ds.iterrows():
        rv = row['run_value_per_100']
        whiff = row['whiff_percent']
        xwoba = row['est_woba']
        pct_rv = ((rv-av_rv_100)/abs(av_rv_100)) * 100
        pct_woba = ((xwoba-av_woba)/av_woba) * 100
        pct_whiff = ((whiff-av_whiff)/av_whiff) * 100
        ls.append((pct_rv + pct_woba + pct_whiff)/3)
    
    return ls
