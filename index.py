import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopip
from PSO import PSO
import sys
import matplotlib.pyplot as plt
'''
seismics = pd.read_csv('earthquakes.csv', usecols = [2,1]).to_numpy()
peruvian_seismics = np.array([s for s in seismics if s[0] < -4.5 or (geopip.search(lng = s[1], lat = s[0]) != None and (geopip.search(lng = s[1], lat = s[0])['NAME'] == 'Peru'))])


for i in range(8):
  pso = PSO(folder = f'0.4-0.5-2.5/{i}', seismics = peruvian_seismics, population = 20, sensors = 52, iterations = 250, w = 0.4, c1 = 0.5, c2 = 2.5, static = True)
  current_sensors, current_fitness, current_time, fitness_for_graph = pso.train()
  with open(f'./new_results/0.4-0.5-2.5_coordinates_{i}', 'w') as file:
    file.write(np.array_str(current_sensors))
  with open(f'./new_results/0.4-0.5-2.5_fitness_{i}', 'w') as file:
    file.write(np.array_str(current_fitness))
  with open(f'./new_results/0.4-0.5-2.5_time_{i}', 'w') as file:
    file.write(np.array_str(current_time))
  with open(f'./new_results/0.4-0.5-2.5_farray_{i}', 'w') as file:
    file.write(np.array_str(fitness_for_graph))
'''
def main(args):
  seismics = pd.read_csv('earthquakes.csv', usecols = [2,1]).to_numpy()
  peruvian_seismics = np.array([s for s in seismics if s[0] < -4.5 or (geopip.search(lng = s[1], lat = s[0]) != None and (geopip.search(lng = s[1], lat = s[0])['NAME'] == 'Peru'))])
  fitnesses = []

  for i in range(8):
    pso = PSO(folder = f'{args[0]}-{args[1]}-{args[2]}/{i}', seismics = peruvian_seismics, population = 20, sensors = 52, iterations = 250, w = float(args[0]), c1 = float(args[1]), c2 = float(args[2]), static = True)
    current_sensors, current_fitness, current_time, fitness_for_graph = pso.train()
    fitnesses.append(fitness_for_graph)
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_coordinates_{i}', 'w') as file:
      file.write(np.array_str(current_sensors))
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_fitness_{i}', 'w') as file:
      file.write(np.array_str(current_fitness))
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_time_{i}', 'w') as file:
      file.write(np.array_str(current_time))
    with open(f'./new_results/{args[0]}-{args[1]}-{args[2]}_farray_{i}', 'w') as file:
      file.write(np.array_str(fitness_for_graph))
  
  x = [0, 10, 20, 30, 40, 50,
      60, 70, 80, 90, 100,
      110, 120, 130, 140, 150,
      160, 170, 180, 190, 200,
      210, 220, 230, 240, 250]
  for i in range(8):
    plt.clf()
    plt.title('Fitness w = {args[0]}, c1 = {args[1]}, c2 = {args[2]}')
    plt.plot(x, fitnesses[0])
    plt.plot(x, fitnesses[1])
    plt.plot(x, fitnesses[2])
    plt.plot(x, fitnesses[3])
    plt.plot(x, fitnesses[4])
    plt.plot(x, fitnesses[5])
    plt.plot(x, fitnesses[6])
    plt.plot(x, fitnesses[7])
    plt.savefig(f'./graphs/f_{args[0]}_{args[1]}_{args[2]}.png')
if __name__ == '__main__':
  main(sys.argv[1:])
