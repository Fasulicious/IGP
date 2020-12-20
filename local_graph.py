import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

class Graph:
  def __init__ (self, folder, seismics, global_best_fitness_collection, global_best_sensors_collection):
    self.m = Basemap(projection = 'mill', llcrnrlat=-20, urcrnrlat = 1, urcrnrlon=-65, llcrnrlon=-85)
    self.folder = folder
    self.seismics = seismics
    self.global_best_fitness_collection = global_best_fitness_collection
    self.global_best_sensors_collection = global_best_sensors_collection
    self.seismic_lons = []
    self.seismic_lats = []
    self.sensor_lons = []
    self.sensor_lats = []

  def graph_coordinates(self):
    for i, (best_sensor, best_fitness) in enumerate(zip(self.global_best_sensors_collection, self.global_best_fitness_collection)):
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
      plt.title(f'Fitness: {best_fitness[-1]}, at iteration: {i * 10}')
      plt.savefig(f'./images/sensors_graphs/{self.folder}.png')

  def graph_fitness(self):
    x = np.linspace(0, 250, 26)
    values = self.folder.split('-')
    plt.clf()
    plt.title(f'Fitness w = {values[0]}, c1 = {values[1]}, c2 = {values[2]}')
    for i in range(8):
      plt.plot(x, self.global_best_fitness_collection[i])  
    plt.savefig(f'./images/fitness_graphs/{self.folder}.png')