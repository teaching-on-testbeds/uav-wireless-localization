import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF

import geopy.distance

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from ipywidgets import interact, fixed, widgets
from mpl_toolkits import mplot3d
plt.rcParams['axes.formatter.useoffset'] = False


BOUND_NE={'lat':35.73030799378120, 'lon':-78.69670002283071}
BOUND_NW={'lat':35.73030799378120, 'lon':-78.69980159100491}
BOUND_SE={'lat':35.72774492720433, 'lon':-78.69670002283071}
BOUND_SW={'lat':35.72774492720433, 'lon':-78.69980159100491}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth
    specified by latitude and longitude.

    Parameters:
        lat1, lon1: Latitude and Longitude of the first point in decimal degrees.
        lat2, lon2: Latitude and Longitude of the second point in decimal degrees.

    Returns:
        Distance in meters between the two points.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Earth radius in meters
    R = 6371000  # meters

    # Differences in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in meters
    distance = R * c

    return distance


def plot_3D(optimizer, elev=20, azim=-70):

  lat_grid = np.linspace(BOUND_SE['lat'], BOUND_NE['lat'], 100)
  lon_grid = np.linspace(BOUND_SE['lon'], BOUND_SW['lon'], 100)

  coord_grid  = np.meshgrid(lat_grid, lon_grid)
  coord_array = np.column_stack([coord_grid[0].ravel(), coord_grid[1].ravel()])

  predictions = optimizer._gp.predict(coord_array, return_std=True)
  gpr_mean = predictions[0].reshape(100, 100)
  gpr_var  = predictions[1].reshape(100, 100)

  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(projection = '3d')

  lat_array = coord_array[:,0].reshape(100, 100)
  lon_array = coord_array[:,1].reshape(100, 100)

  ax.plot_wireframe(lon_array, lat_array, gpr_mean, color='black', alpha=0.3)
  ax.plot_surface(lon_array, lat_array, gpr_mean, cmap=cm.viridis, alpha=0.7)

  y = optimizer._gp.y_train_*optimizer._gp._y_train_std + optimizer._gp._y_train_mean
  ax.scatter3D(optimizer._gp.X_train_[:,1], optimizer._gp.X_train_[:,0], y, c=y, edgecolor='black', s=50);

  ax.ticklabel_format(useOffset=False)
  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")
  ax.set_zlabel("Received Signal Strength")

  ax.view_init(elev=elev, azim=azim)
  plt.show()

def vis_optimizer(optimizer, true_lat, true_lon):

  lat_grid = np.linspace(BOUND_SE['lat'], BOUND_NE['lat'], 100)
  lon_grid = np.linspace(BOUND_SE['lon'], BOUND_SW['lon'], 100)

  coord_grid  = np.meshgrid(lat_grid, lon_grid)
  coord_array = np.column_stack([coord_grid[0].ravel(), coord_grid[1].ravel()])

  predictions = optimizer._gp.predict(coord_array, return_std=True)
  gpr_mean = predictions[0].reshape(100, 100)
  gpr_var  = predictions[1].reshape(100, 100)

  lat_array = coord_array[:,0].reshape(100, 100)
  lon_array = coord_array[:,1].reshape(100, 100)

  try:
    y_max = optimizer._space.target.max()
  except:
    y_max = 0

  fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 8));

  axes[0, 0].contourf(lon_array, lat_array, gpr_mean, cmap='viridis', vmin=20, vmax=50, levels=8);
  axes[0, 0].set_title("Mean of predicted received signal strength");

  #axes[0, 1].imshow(img, interpolation='nearest', aspect='auto', extent=[BOUND_SW['lon'], BOUND_SE['lon'],BOUND_SE['lat'], BOUND_NE['lat']])
  axes[0, 1].set_title("Search area");
  try:
    for i, x_train in enumerate(optimizer._gp.X_train_):
      axes[0, 1].text(x_train[1], x_train[0], str(i), color="white", fontsize=8, fontweight='heavy', ha="center", va="center",
             bbox = dict(boxstyle=f"circle", fc="black"))
  except AttributeError:
    pass


  axes[1, 0].contourf(lon_array, lat_array, gpr_var, cmap='viridis');
  axes[1, 0].set_title("Variance of predicted received signal strength");

  try:
    util  = optimizer.acquisition_function._get_acq(gp=optimizer._gp)(coord_array).reshape(100, 100)
    axes[1, 1].contourf(lon_array, lat_array, util, cmap='viridis_r', vmin=-50, vmax=-20, levels=8);
  except AttributeError:
    pass
  axes[1, 1].set_title("Value of utility function");

  try:
    for i, x_train in enumerate(optimizer._gp.X_train_):
      axes[0, 0].plot(x_train[1], x_train[0], marker='P', markersize=10, markerfacecolor='black', markeredgecolor='white');
      axes[1, 0].plot(x_train[1], x_train[0], marker='P', markersize=10, markerfacecolor='black', markeredgecolor='white');
      axes[1, 1].plot(x_train[1], x_train[0], marker='P', markersize=10, markerfacecolor='black', markeredgecolor='white');
  except AttributeError:
    pass

  axes[0, 0].plot(true_lon, true_lat, marker='*', color='red', markersize=15)
  axes[0, 1].plot(true_lon, true_lat, marker='*', color='red', markersize=15)
  axes[1, 0].plot(true_lon, true_lat, marker='*', color='red', markersize=15)
  axes[1, 1].plot(true_lon, true_lat, marker='*', color='red', markersize=15)

  try:
      n_samples = optimizer._gp.X_train_.shape[0]
  except AttributeError:
      n_samples = 0

  true_pos = (true_lat, true_lon)
  est_pos  = (coord_array[np.argmax(gpr_mean)][0], coord_array[np.argmax(gpr_mean)][1])
  est_error = haversine_distance(true_lat, true_lon, est_pos[0], est_pos[1])
  plt.suptitle("Optimizer state after %d samples, error: %f m" % (n_samples, est_error));

  axes[0, 0].plot(est_pos[1], est_pos[0], marker='X', markersize=15, markerfacecolor='black', markeredgecolor='white');

  plt.ticklabel_format(useOffset=False) ;
  plt.tight_layout();


def est_error(optimizer, true_lat, true_lon):

  lat_grid = np.linspace(BOUND_SE['lat'], BOUND_NE['lat'], 100)
  lon_grid = np.linspace(BOUND_SE['lon'], BOUND_SW['lon'], 100)

  coord_grid  = np.meshgrid(lat_grid, lon_grid)
  coord_array = np.column_stack([coord_grid[0].ravel(), coord_grid[1].ravel()])

  predictions = optimizer._gp.predict(coord_array, return_std=False)
  gpr_mean = predictions.reshape(100, 100)

  est_pos  = (coord_array[np.argmax(gpr_mean)][0], coord_array[np.argmax(gpr_mean)][1])
  est_error = haversine_distance(true_lat, true_lon, est_pos[0], est_pos[1])

  return est_error

def plot_position_error_over_time(df, true_lat, true_lon):
    df = df.copy()
    df['time_sec'] = pd.to_timedelta(df['timestamp']).dt.total_seconds()
    df['distance_m'] = df.apply(
        lambda row: haversine_distance(true_lat, true_lon, row['best_lat'], row['best_lon']),
        axis=1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(df['time_sec'], df['distance_m'], marker='o', linestyle='-')
    plt.title('Distance Between True and Best Position vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (meters)')
    plt.ylim(bottom=0)  

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

    plt.grid(True)
    plt.tight_layout()
    plt.show()
