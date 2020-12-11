import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopip
from PSO import PSO

seismics = pd.read_csv('earthquakes.csv', usecols = [2,1]).to_numpy()
peruvian_seismics = np.array([s for s in seismics if s[0] < -4.5 or (geopip.search(lng = s[1], lat = s[0]) != None and (geopip.search(lng = s[1], lat = s[0])['NAME'] == 'Peru'))])


for i in range(8):
  pso = PSO(folder = f'0.4-0.5-2.5/{i}', seismics = peruvian_seismics, population = 20, sensors = 52, iterations = 250, w = 0.4, c1 = 0.5, c2 = 2.5, static = True)
  current_sensors, current_fitness, current_time = pso.train()
  with open(f'./new_results/0.4-0.5-2.5_coordinates_{i}', 'w') as file:
    file.write(np.array_str(current_sensors))
  with open(f'./new_results/0.4-0.5-2.5_fitness_{i}', 'w') as file:
    file.write(np.array_str(current_fitness))
  with open(f'./new_results/0.4-0.5-2.5_time_{i}', 'w') as file:
    file.write(np.array_str(current_time))
