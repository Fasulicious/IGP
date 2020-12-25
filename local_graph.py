import matplotlib.pyplot as plt
import numpy as np
import geopip
from mpl_toolkits.basemap import Basemap

class Graph:
  def __init__ (self, folder, seismics):
    self.m = Basemap(projection = 'mill', llcrnrlat=-20, urcrnrlat = 1, urcrnrlon=-65, llcrnrlon=-85)
    self.folder = folder
    self.seismics = seismics
    self.seismic_lons = []
    self.seismic_lats = []
    self.sensor_lons = []
    self.sensor_lats = []
  
  def get_distance(self, P, Q):
    fiP = P[0] * np.pi / 180
    fiQ = Q[0] * np.pi / 180
    dfi = fiQ - fiP
    dlm = (Q[1] - P[1]) * np.pi / 180
    a = np.sin(dfi / 2) * np.sin(dfi / 2) + np.cos(fiP) * np.cos(fiQ) * np.sin(dlm / 2) * np.sin(dlm / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1- a))
    return 6371 * c

  def old_fitness(self, sensors):
    total_distance = 0
    sensors_inside_peru = 0
    for seismic in self.seismics:
      distances = []
      for sensor in sensors:
        distances.append(self.get_distance(seismic, sensor))
      distances = np.array(distances)
      distances = np.sort(distances)
      total_distance += np.sum(distances[0: 3])
    for sensor in sensors:
      geolocation = geopip.search(lng = sensor[1], lat = sensor[0])
      if geolocation != None and geolocation['NAME'] == 'Peru':
        sensors_inside_peru += 1
    current_fitness = total_distance * 2 / (1 + sensors_inside_peru / len(sensors))
    return current_fitness

  def graph_coordinates(self, global_best_sensors_collection, current_fitness, iter):
    for i, (best_sensor, best_fitness) in enumerate(zip(global_best_sensors_collection, current_fitness)):
      old_fitness = self.old_fitness(best_sensor)
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
      for lat, lon in best_sensor:
        self.sensor_lats.append(lat)
        self.sensor_lons.append(lon)
      m_sensor_lons, m_sensor_lats = self.m(self.sensor_lons, self.sensor_lats)
      self.m.scatter(m_seismic_lons, m_seismic_lats, marker = 'o', color = 'b', zorder = 5)
      self.m.scatter(m_sensor_lons, m_sensor_lats, marker = 'X', color = 'r', zorder = 10)
      plt.title(f'Fitness: {best_fitness}, Old Fitness: {old_fitness}, at iteration: {i * 10}')
      plt.savefig(f'./images/sensors_graphs/{self.folder}_{iter}_{i}.png')
  
  def graph_fitness(self, global_best_fitness_collection):
    x = np.linspace(0, 250, 26)
    values = self.folder.split('-')
    plt.clf()
    plt.title(f'Fitness w = {values[0]}, c1 = {values[1]}, c2 = {values[2]}')
    for i in range(8):
      plt.plot(x, global_best_fitness_collection[i])  
    plt.savefig(f'./images/fitness_graphs/{self.folder}.png')