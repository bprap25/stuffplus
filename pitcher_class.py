from pybaseball import statcast
from pybaseball import pitching_stats
from pybaseball import cache
from datetime import date

class Pitcher():
    def __init__(self, name):
        self.name = name
        self.pitches = dict()
        self.num_pitches = 1
    
    def add_pitch(self, pitch_type, metrics):
        """
        Method used when new pitch is detected
        Args: pitch_type: str (FB, SL, CH, ...)
        metrics: list [speed, release_posx, release_posz, rel_posy, p_throws, pfx_x, pfx_z, spin_axis, spin_rate]
        """
        self.pitches[pitch_type] = metrics
        self.pitches[pitch_type].append(1) #initialize first pitch

    def update_pitch(self, pitch_type, metrics):
        metrics.append(1)
        self.pitches[pitch_type] = [x + y for x,y in zip(metrics,self.pitches[pitch_type])] #keeps summation, takes averages in batch_averages
        self.num_pitches += 1

    def batch_averages(self):
        """
        After reading all data, gets averages of each pitch"""
        for pitch in self.pitches.keys():
            num_type = self.pitches[pitch][-1]
            new_pitch_stats = [self.pitches[pitch][x]/num_type for x in range(len(self.pitches[pitch]))]
            new_pitch_stats.append(num_type/self.num_pitches)
            self.pitches = new_pitch_stats
    
