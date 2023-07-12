import torch
from DataPull import get_data, pitcher_class
import time

start_time = time.time()
pt_data = get_data.get_pitcher_data()
p_dict = pitcher_class.format_pitcher_data(pt_data)
new_ds = pitcher_class.process_batch_avg(p_dict)
print(new_ds)
print("--- %s seconds ---" % (time.time() - start_time))