import sys
import numpy as np
from local_graph import Graph

def main(args):
  fitness = []
  for i in range(8):
    with open(f'./results/{args[0]}-{args[1]}-{args[2]}_fitness_{i}') as file:
      fitness_string = file.read()
    fitness.append(np.fromstring(fitness_string[1:-1], dtype = np.float64, sep = ' '))
  fitness = np.array(fitness)
  g = Graph(folder = f'{args[0]}-{args[1]}-{args[2]}', seismics = [], global_best_fitness_collection = fitness, global_best_sensors_collection = [])
  g.graph_fitness()

if __name__ == '__main__':
  main(sys.argv[1:])