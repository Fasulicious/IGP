import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopip
import numpy as np
import time
np.seterr(divide='ignore', invalid='ignore')

class PSO:
  def __init__ (self, folder, seismics, population, sensors, iterations, w = None, c1 = None, c2 = None, sigma = 0.25, static = False):
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
    self.folder = folder
    # CAMBIO PARA TIPO DE INICIALIZACION DE SENSORES
    #self.x = np.array([np.hstack((np.random.uniform(-18, -3, (sensors, 1)), np.random.uniform(-82, -69, (sensors, 1))))  for _ in range(population)])
    self.x = np.array([self.init_x() for _ in range(population)])
    self.v = np.zeros((population, sensors, 2))
    self.global_best_fitness = np.inf
    self.global_best_sensors = []
    self.best_fitness = [np.inf for _ in range(population)]
    self.best_sensors = self.x.copy()
    self.m = Basemap(projection = 'mill', llcrnrlat=-20, urcrnrlat = 1, urcrnrlon=-65, llcrnrlon=-85)
    self.seismic_lons = []
    self.seismic_lats = []
    self.sensor_lons = []
    self.sensor_lats = []

  # UBICACION DE SENSORES DENTRO DEL PERU
  def init_x(self):
    temp = []
    while len(temp) < self.sensors:
      location = [np.random.uniform(-3, -18), np.random.uniform(-69, -82)]
      geolocation = geopip.search(lng = location[1], lat = location[0])
      if geolocation != None and geolocation['NAME'] == 'Peru':
        temp.append(location)
    return temp
  
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
    return geolocation != None and geolocation['NAME'] == 'Peru'

  def get_seismic_distance(self, distances):
    current_distances = np.array(distances)
    current_distances = np.sort(current_distances)
    return np.sum(current_distances[0: 3])

  # CALCULO DEL FITNESS
  def fitness(self):
    start = time.time()
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
    end = time.time()
    print(end - start)
    print(fitness)
    return fitness

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
    start = time.time()
    best_fitness_for_graph = []
    for i in range(self.iterations):
      current_iteration = i + 1
      print(f'Iteration {i+1} before get the fitness')
      # CALCULAR FITNESS Y ACTUALIZAR
      fitness = self.fitness()
      print(f'Iteration {i+1} after get the fitness')
      self.update(fitness)
      # DIBUJAR CADA 50 ITERACIONES
      if i % 10 == 0:
        plt.clf()
        plt.figure(figsize=(24, 12))
        self.m.drawcoastlines()
        self.m.drawcountries(linewidth = 1)
        self.seismic_lons.clear()
        self.seismic_lats.clear()
        self.sensor_lats.clear()
        self.sensor_lons.clear()
        for lat, lon in self.seismics:
          self.seismic_lats.append(lat)
          self.seismic_lons.append(lon)
        m_seismic_lons, m_seismic_lats = self.m(self.seismic_lons, self.seismic_lats)
        for lat, lon in self.global_best_sensors:
          self.sensor_lats.append(lat)
          self.sensor_lons.append(lon)
        m_sensor_lons, m_sensor_lats = self.m(self.sensor_lons, self.sensor_lats)
        self.m.scatter(m_seismic_lons, m_seismic_lats, marker = 'o', color = 'b', zorder = 5)
        self.m.scatter(m_sensor_lons, m_sensor_lats, marker = 'X', color = 'r', zorder = 10)
        plt.title(f'Fitness: {self.global_best_fitness}, at iteration: {i}')
        plt.savefig(f'./new_images/{self.folder}/iteration_{i}.png')
        best_fitness_for_graph.append(self.global_best_fitness)
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
    end = time.time()
  # DIBUJAR MEJOR ESTADO FINAL
    plt.clf()
    plt.figure(figsize=(24, 12))
    self.m.drawcoastlines()
    self.m.drawcountries(linewidth = 1)
    self.seismic_lons.clear()
    self.seismic_lats.clear()
    self.sensor_lats.clear()
    self.sensor_lons.clear()
    for lat, lon in self.seismics:
      self.seismic_lats.append(lat)
      self.seismic_lons.append(lon)
    m_seismic_lons, m_seismic_lats = self.m(self.seismic_lons, self.seismic_lats)
    for lat, lon in self.global_best_sensors:
      self.sensor_lats.append(lat)
      self.sensor_lons.append(lon)
    m_seismic_lons, m_seismic_lats = self.m(self.seismic_lons, self.seismic_lats)
    m_sensor_lons, m_sensor_lats = self.m(self.sensor_lons, self.sensor_lats)
    self.m.scatter(m_seismic_lons, m_seismic_lats, marker = 'o', color = 'b', zorder = 5)
    self.m.scatter(m_sensor_lons, m_sensor_lats, marker = 'X', color = 'r', zorder = 10)
    plt.title(f'Fitness: {self.global_best_fitness}, at iteration: {i + 1}')
    plt.savefig(f'./new_images/{self.folder}/iteration_{i}.png')
    return np.array(self.global_best_sensors), np.array(self.global_best_fitness), np.array(end - start), np.array(best_fitness_for_graph)
    