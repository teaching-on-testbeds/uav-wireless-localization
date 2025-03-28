import asyncio
import math
import datetime
import csv
import pickle

from typing import List
from struct import unpack
from argparse import ArgumentParser


from aerpawlib.runner import StateMachine
from aerpawlib.vehicle import Vehicle, Drone
from aerpawlib.runner import state, timed_state
from aerpawlib.util import Coordinate, VectorNED
from aerpawlib.safetyChecker import SafetyCheckerClient

from radio_power import RadioEmitter

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF

BOUND_NE={'lat':35.73030799378120, 'lon':-78.69670002283071}
BOUND_NW={'lat':35.73030799378120, 'lon':-78.69980159100491}
BOUND_SE={'lat':35.72774492720433, 'lon':-78.69670002283071}
BOUND_SW={'lat':35.72774492720433, 'lon':-78.69980159100491}

MAX_LON = BOUND_NE['lon']
MIN_LON = BOUND_NW['lon']
MAX_LAT = BOUND_NE['lat']
MIN_LAT = BOUND_SE['lat']

SEARCH_ALTITUDE = 50 # in meters

SEARCH_RESOLUTION = 500
COORD_GRID  = np.meshgrid(np.linspace(BOUND_SE['lat'], BOUND_NE['lat'], 500), 
                          np.linspace(BOUND_SE['lon'], BOUND_SW['lon'], 500)
                          )
COORD_ARRAY = np.column_stack([COORD_GRID[0].ravel(), COORD_GRID[1].ravel()])


class RoverSearch(StateMachine):
    best_measurement = float("-inf")
    best_pos = Coordinate((MIN_LAT + MAX_LAT)/2, (MIN_LON + MAX_LON), SEARCH_ALTITUDE)

    start_time = None
    search_time = None

    heading_seq_n_idx = 0
    heading_seq_w_idx = 0

    measurement_list = []

    # start in the middle of the search space
    next_waypoint = {'lat': (MIN_LAT + MAX_LAT)/2, 'lon': (MIN_LON + MAX_LON)/2}

    # change this utility function or its arguments to manage the 
    # exploration/exploitation tradeoff
    utility = acquisition.UpperConfidenceBound()

    optimizer = BayesianOptimization(
      f=None,
      pbounds={'lat': (MIN_LAT, MAX_LAT), 'lon': (MIN_LON, MAX_LON)},
      verbose=0,
      random_state=0,
      allow_duplicate_points=True,
      acquisition_function = utility
    )
    # set the kernel
    kernel = RBF()
    optimizer._gp.set_params(kernel = kernel)

    def initialize_args(self, extra_args: List[str]):
        """Parse arguments passed to vehicle script"""
        # current time string:
        dt_str =  datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Default output files for search data
        defaultFile = "/root/Results/ROVER_SEARCH_DATA_%s.csv" %  dt_str        

        parser = ArgumentParser()
        parser.add_argument("--safety_checker_ip", help="ip of the safety checker server")
        parser.add_argument("--safety_checker_port", help="port of the safety checker server")
        parser.add_argument(
            "--log",
            help="File to record search path and RSSI values",
            required=False,
            default=defaultFile,
        )
        parser.add_argument(
            "--save_csv",
            help="Whether to save a file with the search path and RSSI values",
            default=True,
        )
        parser.add_argument(
            "--search_time",
            help="How long in minutes to search for the rover",
            required=False,
            default=5,
            type=int,
        )

        args = parser.parse_args(args=extra_args)

        self.safety_checker = SafetyCheckerClient(args.safety_checker_ip, args.safety_checker_port)
        self.log_file = open(args.log, "w+")
        self.save_csv = args.save_csv
        self.search_time = datetime.timedelta(minutes=args.search_time)

        self.pickle_file_name = "/root/Results/opt_final_%s.pickle" % dt_str

        # Create a CSV writer object if we are saving data
        if self.save_csv:
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(["timestamp","longitude", "latitude", "altitude", "RSSI", "best_lat", "best_lon", "best_RSSI"])
        
    @state(name="start", first=True)
    async def start(self, vehicle: Drone):
        # record the start time of the search
        await vehicle.takeoff(SEARCH_ALTITUDE)
        self.start_time = datetime.datetime.now()
        print("Reached altitude, starting search.")

        return "probe"


    @state(name="register")
    async def register(self, vehicle: Drone):

        # register last value
        self.optimizer.register(params={'lat': vehicle.position.lat, 'lon': vehicle.position.lon}, target=self.measurement_list[-1]['power'])

        # get the prediction of the model - where will the target be highest?
        predictions = self.optimizer._gp.predict(COORD_ARRAY, return_std=False)
        gpr_mean = predictions.reshape(SEARCH_RESOLUTION, SEARCH_RESOLUTION)

        self.best_measurement = gpr_mean.max()
        self.best_pos = Coordinate(COORD_ARRAY[np.argmax(gpr_mean)][0],COORD_ARRAY[np.argmax(gpr_mean)][1], SEARCH_ALTITUDE)

        # save the measurements and estimates if logging to file
        if self.best_pos and self.save_csv:
            for m in self.measurement_list:
                self.csv_writer.writerow(
                    [
                        datetime.datetime.now() - self.start_time, # elapsed search time
                        self.measurement_list[-1]['lat'], # current position - latitude
                        self.measurement_list[-1]['lon'], # current position - longitude
                        SEARCH_ALTITUDE,                  # altitude
                        self.measurement_list[-1]['power'], # measurement of received signal power at this position
                        self.best_pos.lat, # estimated best position - latitude
                        self.best_pos.lon, # estimated best position - longitude
                        self.best_measurement # estimate of mean received signal power at estimated best position
                    ]
                )

        # reset measurement list
        self.measurement_list = []
        return "suggest"
    

    @state(name="suggest")
    async def suggest(self, vehicle: Drone):
        # suggest the next waypoint to visit
        self.next_waypoint = self.optimizer.suggest()

        return "probe"
    

    @state(name="probe")
    async def probe(self, vehicle: Drone):
        # stop if search time is over
        if datetime.datetime.now() - self.start_time > self.search_time:
            return "end"

        # go to the waypoint
        next_pos =  Coordinate(
            np.clip(self.next_waypoint['lat'], MIN_LAT, MAX_LAT),
            np.clip(self.next_waypoint['lon'], MIN_LON, MAX_LON),
            SEARCH_ALTITUDE)
        
        moving = asyncio.ensure_future(
            vehicle.goto_coordinates(next_pos)
        )
        
        while not moving.done():
            await asyncio.sleep(0.2)

        # take a radio measurement
        # Open data buffer
        f = open("/root/Power", "rb")
        # unpack binary reading into a float
        measurement_from_file = unpack("<f", f.read(4))
        measurement = measurement_from_file[0]
        # close the data buffer
        f.close()

        pos = vehicle.position
        self.measurement_list.append( 
            {'lat': pos.lat, 'lon': pos.lon, 'power': measurement} 
            )
                
        return "register"


    @state(name="end")
    async def end(self, vehicle: Drone):
        # Return vehicle to start and land it

        print(
            f"Search time of {self.search_time} minutes has elapsed. Returning to launch!"
        )
        print(
            f"Best rover location estimate {self.best_pos.lat, self.best_pos.lon} with measurement {self.best_measurement} after {datetime.datetime.now()-self.start_time} minutes"
        )
      
        # save the final optimizer
        with open(self.pickle_file_name, 'wb') as handle:
            pickle.dump(self.optimizer, handle)

        # go home
        home_coords = Coordinate(
            vehicle.home_coords.lat, vehicle.home_coords.lon, vehicle.position.alt
        )
        await vehicle.goto_coordinates(home_coords)
        await vehicle.land()
        print("Done!")

