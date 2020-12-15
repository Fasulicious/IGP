import geopip
import numpy as np
from time import time
np.seterr(divide='ignore', invalid='ignore')

class PSO:
  def __init__ (self, seismics, population, sensors, iterations, w = None, c1 = None, c2 = None, sigma = 0.25, static = False):
    self.seismics = seismics
    self.population = population
    self.sensors = sensors
    self.iterations = iterations
    self.w = w
    self.w_min = 0.4
    self.w_max = 0.9
    self.c1 = c1
    self.c1_min = 0.5
    self.c1_max = 2.5
    self.c2 = c2
    self.c2_min = 0.5
    self.c2_max = 2.5
    self.sigma = sigma
    self.static = static
    self.x = np.array([self.init_x() for _ in range(population)])
    self.v = np.zeros((population, sensors, 2))
    self.global_best_fitness = np.inf
    self.global_best_sensors = []
    self.global_best_fitness_collection = []
    self.global_best_sensors_collection = []
    self.best_fitness = [np.inf for _ in range(population)]
    self.best_sensors = self.x.copy()
    

  # UBICACION DE SENSORES DENTRO DEL PERU
  def init_x(self):
    temp = []
    while len(temp) < self.sensors:
      location = [np.random.uniform(-3, -18), np.random.uniform(-69, -82)]
      geolocation = geopip.search(lng = location[1], lat = location[0])
      if geolocation != None and geolocation['NAME'] == 'Peru':
        temp.append(np.array(location))
    return np.array(temp)
  
  # OBTENER DISTANCIA REAL SOBRE LA TIERRA
  def get_distance(self, P, Q):
    fiP = P[0] * np.pi / 180
    fiQ = Q[0] * np.pi / 180
    dfi = fiQ - fiP
    dlm = (Q[1] - P[1]) * np.pi / 180
    a = np.sin(dfi / 2) * np.sin(dfi / 2) + np.cos(fiP) * np.cos(fiQ) * np.sin(dlm / 2) * np.sin(dlm / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1- a))
    return 6371 * c

  
  def check_inside_peru(self, sensor):
    geolocation = geopip.search(lng = sensor[1], lat = sensor[0])
    return 3 if geolocation != None and geolocation['NAME'] == 'Peru' else 1
  
  def get_multiplier(self, sensors):
    #return np.array([self.check_inside_peru(sensor) for sensor in sensors])
    return np.fromiter((self.check_inside_peru(sensor) for sensor in sensors), np.int8)

  def get_seismic_distance(self, distances):
    current_distances = np.array(distances)
    current_distances = np.sort(current_distances)
    return np.sum(current_distances[0: 3])
  
  def seismic_to_sensors(self, seismic, sensors):
    multiplier = self.get_multiplier(sensors)
    return np.fromiter((self.get_distance(seismic, sensor) for sensor in sensors), np.float32) * multiplier
  
  def get_distances(self, sensors, seismics):
    #return np.fromiter((self.seismic_to_sensors(seismic, sensors) for seismic in seismics), np.float32)
    distances = np.sort(np.array(np.array([self.seismic_to_sensors(seismic, sensors) for seismic in seismics])))
    return np.sum(distances[0:3])

  # CALCULO DEL FITNESS
  def fitness(self):
    s = time()
    print(np.sum(np.array([self.get_distances(sensors, self.seismics) for sensors in self.x])))
    #print(self.get_distances(self.x[0], self.seismics))
    #print(self.seismic_to_sensors(self.seismics[0], self.x[0]))
    e = time()
    print(e - s)
    '''
    fitness = [
      np.sum(
        [
          self.get_seismic_distance(
            [
              self.get_distance(seismic, sensor)
              if self.check_inside_peru(sensor)
              else 3 * self.get_distance(seismic, sensor)
              for sensor
              in sensors
            ]
          )
          for seismic
          in self.seismics
        ]
      )
      for sensors
      in self.x
    ]
    return fitness
    '''
  # SE ACTUALIZA MEJOR GLOBAL, MEJOR PERSONAL
  def update(self, fitness):
    # MEJOR GLOBAL
    if min(fitness) < self.global_best_fitness:
      self.global_best_fitness = min(fitness)
      self.global_best_sensors = self.x[fitness.index(min(fitness))]
    # MEJOR PERSONAL
    for i, (best, current) in enumerate(zip(self.best_fitness, fitness)):
      if current < best:
        self.best_fitness[i] = fitness[i]
        self.best_sensors[i] = self.x[i]

  # ENTRENAMIENTO
  def train(self):
    for i in range(self.iterations):
      current_iteration = i + 1
      print(f'Iteration {i+1} before get the fitness')
      # CALCULAR FITNESS Y ACTUALIZAR
      fitness = self.fitness()
      print(f'Iteration {i+1} after get the fitness')
      self.update(fitness)
      # AGREGAR CADA 10 ITERACIONES
      if i % 10 == 0:
        self.global_best_fitness_collection.append(self.global_best_fitness)
        self.global_best_sensors_collection.append(self.global_best_sensors)
      # ACTUALIZACION DE PARAMETROS SI ES AUTOAJUSTE
      w = self.w if self.static else self.w_max + (self.w_min - self.w_max) * (current_iteration - 1) / (self.iterations - 1)
      c1 = self.c1 if self.static else self.c1_max + (self.c1_min - self.c1_max) * current_iteration / self.iterations
      c2 = self.c2 if self.static else self.c2_min + (self.c2_max - self.c2_min) * current_iteration / self.iterations
      # NORMA DEL VECTOR POSICION ACTUAL A MEJOR PERSONAL
      norma_p = np.reshape(np.linalg.norm(self.best_sensors - self.x, axis = 2), [self.population, self.sensors, 1])
      # VECTOR UNITARIO DE POSICION ACTUAL A MEJOR PERSONAL
      unit_p = (self.best_sensors - self.x) / norma_p
      # RANDOM EN DIRECCTION POSICION ACTUAL A MEJOR PERSONAL
      rand_p = np.random.normal(norma_p / 2, self.sigma)
      # CONTRIBUCION INDIVIDUAL AL MOVIMIENTO
      individual = c1 * rand_p * unit_p
      individual[np.isnan(individual)] = 0
      # RESHAPE DEL MEJOR GLOBAL PARA CADA FAMILIA DE SENSORES
      g = np.repeat(self.global_best_sensors[np.newaxis, ...], self.population, axis = 0)
      # NORMA DEL VECTOR POSICION ACTUAL A MEJOR GLOBAL
      norma_g = np.reshape(np.linalg.norm(g - self.x, axis = 2), [self.population, self.sensors, 1])
      # VECTOR UNITARIO DE POSICION ACTUAL A MEJOR GLOBAL
      unit_g = (g - self.x) / norma_g
      # RANDOM EN DIRECCTION POSICION ACTUAL A MEJOR GLOBAL
      rand_g = np.random.normal(norma_g / 2, self.sigma)
      # CONTRIBUCION COLECTIVA AL MOVIMIENTO
      colectivo = c2 * rand_g * unit_g
      colectivo[np.isnan(colectivo)] = 0
      # NUEVA VELOCIDAD CONTRIBUCIONES DE INTERCIA, INDIVIDUAL Y COLECTIVO
      self.v = w * self.v + individual + colectivo
      self.x = self.x + self.v
  # AGREGAR ESTADO FINAL Y RETORNAR
    self.global_best_fitness_collection.append(self.global_best_fitness)
    self.global_best_sensors_collection.append(self.global_best_sensors)
    return np.array(self.global_best_sensors_collection), np.array(self.global_best_fitness_collection)
    