import sys
import pandas as pd
import numpy as np
import geopip
from local_graph import Graph

def main(args):
  fitness = []
  seismics = pd.read_csv('earthquakes.csv', usecols = [2,1]).to_numpy()
  peruvian_seismics = np.array([s for s in seismics if s[0] < -4.5 or (geopip.search(lng = s[1], lat = s[0]) != None and (geopip.search(lng = s[1], lat = s[0])['NAME'] == 'Peru'))])
  for i in range(8):
    with open(f'./results/{args[0]}-{args[1]}-{args[2]}_fitness_{i}', 'rb') as file:
      current_fitness = np.load(file)
    fitness.append(current_fitness)
    with open(f'./results/{args[0]}-{args[1]}-{args[2]}_coordinates_{i}', 'rb') as file:
      coordinates = np.load(file)
      g = Graph(folder = f'{args[0]}-{args[1]}-{args[2]}', seismics = peruvian_seismics)
      g.graph_coordinates(coordinates, current_fitness, i)
      
  g = Graph(folder = f'{args[0]}-{args[1]}-{args[2]}', seismics = peruvian_seismics)
  g.graph_fitness(fitness)
  
    

if __name__ == '__main__':
  main(sys.argv[1:])