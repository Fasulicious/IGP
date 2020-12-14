import matplotlib.pyplot as plt
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

  def graph(self):
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
      plt.title(f'Fitness: {best_fitness}, at iteration: {i * 10}')
      plt.savefig(f'./new_images/{self.folder}/iteration_{i * 10}.png')

'''
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
'''