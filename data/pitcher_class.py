from pybaseball import statcast
from pybaseball import pitching_stats
from pybaseball import cache
from datetime import date
import pandas as pd
# from get_data import get_pitcher_data

class Pitcher():
    def __init__(self, name, hand):
        self.name = name
        self.throws = hand
        self.pitches = dict()
        self.num_pitches = 0
    
    def add_pitch(self, pitch_type, metrics):
        """
        Method used when new pitch is detected
        Args: pitch_type: str (FB, SL, CH, ...)
        metrics: list [speed, release_posx, release_posz, rel_posy, p_throws, pfx_x, pfx_z, spin_axis, spin_rate]
        """
        self.pitches[pitch_type] = metrics
        self.pitches[pitch_type].append(1) #initialize first pitch
        self.num_pitches += 1

    def update_pitch(self, pitch_type, metrics):
        metrics.append(1)
        self.pitches[pitch_type] = [x + y for x,y in zip(metrics,self.pitches[pitch_type])] #keeps summation, takes averages in batch_averages
        self.num_pitches += 1

    def batch_averages(self):
        """
        After reading all data, gets averages of each pitch
        """
        for pitch in self.pitches.keys():
            metrics_type = self.pitches[pitch]
            num_type = metrics_type[-1]
            new_pitch_stats = []
            for x in range(len(metrics_type) - 1):
                metric = metrics_type[x]
                new_pitch_stats.append(metric/num_type)
            new_pitch_stats.append(num_type/self.num_pitches)
            self.pitches[pitch] = new_pitch_stats
    
def format_pitcher_data(pt_data):
    pitcher_dict = dict() #maps string rep of pitcher name to pitcher object
    for index, row in pt_data.iterrows():
        pitcher_name = row['player_name']
        pitch_type = row['pitch_name']
        pitcher_throws = row['p_throws']
        metrics = [row['release_speed'], row['release_pos_x'], row['release_pos_z'], row['pfx_x'],
                   row['pfx_z'], row['release_spin_rate'], row['release_extension'], row['spin_axis']]
        if pitcher_name not in pitcher_dict.keys():
            init_pitcher = Pitcher(pitcher_name, pitcher_throws)
            pitcher_dict[pitcher_name] = init_pitcher
            init_pitcher.add_pitch(pitch_type, metrics)
        else:
            pitcher_obj = pitcher_dict[pitcher_name]
            if pitch_type in pitcher_obj.pitches.keys():
                pitcher_obj.update_pitch(pitch_type, metrics)
            else:
                pitcher_obj.add_pitch(pitch_type, metrics)
    
    return pitcher_dict

def process_batch_avg(p_dict):
    avg_df = pd.DataFrame(columns = ['player_name', 'pitch_name', 'p_throws','release_speed','release_pos_x','release_pos_z','pfx_x',
                       'pfx_z','release_spin_rate','release_extension','spin_axis', 'useage %'])
    for pitcher in p_dict.values():
        name = pitcher.name
        pitcher.batch_averages()
        for pitch in pitcher.pitches:
            ls = [name] + [pitch] + [pitcher.throws] + pitcher.pitches[pitch]
            avg_df.loc[len(avg_df)] = ls
    return avg_df