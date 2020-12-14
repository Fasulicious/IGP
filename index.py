import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopip
from PSO import PSO
from local_graph import Graph
import sys

def main(args):
  seismics = pd.read_csv('earthquakes.csv', usecols = [2,1]).to_numpy()
  peruvian_seismics = np.array([s for s in seismics if s[0] < -4.5 or (geopip.search(lng = s[1], lat = s[0]) != None and (geopip.search(lng = s[1], lat = s[0])['NAME'] == 'Peru'))])
  fitnesses = []

  for i in range(8):
    pso = PSO(seismics = peruvian_seismics, population = 20, sensors = 52, iterations = 250, w = float(args[0]), c1 = float(args[1]), c2 = float(args[2]), static = True)
    global_best_sensors_collection, global_best_fitness_collection = pso.train()
    fitnesses.append(global_best_fitness_collection)
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_coordinates_{i}', 'w') as file:
      file.write(np.array_str(global_best_sensors_collection))
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_fitness_{i}', 'w') as file:
      file.write(np.array_str(global_best_fitness_collection))
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_farray_{i}', 'w') as file:
      file.write(np.array_str(fitnesses))

  g = Graph()

if __name__ == '__main__':
  main(sys.argv[1:])
