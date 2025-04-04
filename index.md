
# UAV-assisted wireless localization
 
You've been asked to contribute your machine learning expertise to a
crucial and potentially life-saving mission.

A pair of hikers has gone missing in a national park, and they are
believed to be critically injured. Fortunately, they have activated a
wireless locator beacon, and it transmits a constant wireless signal
from their location. Unfortunately, their beacon was not able to get a
satellite fix, so their GPS position is not known.

To rescue the injured hikers, therefore, their location must be
estimated using the signal strength of the wireless signal from the
beacon: when a radio receiver is close to the beacon, the signal
strength will be high. When a radio receiver is far from the beacon, the
signal strength will be lower. (The relationship is noisy, however; the
wireless signal also fluctuates over time, even with a constant
distance.)

You are going to fly an unmanned aerial vehicle (UAV) with a radio
receiver around the area where they were last seen, and use the received
wireless signal strength to fit a machine learning model that will
estimate the hikers' position. Then, you'll relay this information to
rescuers, who will try to reach that position by land. (Unfortunately,
due to dense tree cover, the UAV will not be able to visually confirm
their position.)

There is a complication, though - the UAV has a limited battery life,
and therefore, limited flight time. You'll have to get an accurate
estimate of the hikers' position in a very short time!

------------------------------------------------------------------------

#### Objectives

In this experiment, you will:

-   observe how the Gaussian Process Regression approximates the true function 
    of signal strength vs. position, in order to find the position where signal 
    strength will be maximized.
-   observe how the kernel is used in a Gaussian Process Regression, and controls
    the shape of the learned function.
-   observe how Bayesian Optimization is used to dynamically decide which training
    data point to acquire next.

------------------------------------------------------------------------

#### Prerequisites

To complete this assignment, you should already have an account on
AERPAW with the experimenter role, be part of a project, have all the
necessary software to work with AERPAW experiments. You should also have
already created an experiment with one UAV and one UGV. 
(See: [Hello, AERPAW](https://teaching-on-testbeds.github.io/hello-aerpaw/)) 

------------------------------------------------------------------------

#### Citations

This experiment uses the Bayesian Optimization implementation of

> Fernando Nogueira, "Bayesian Optimization: Open source constrained
> global optimization tool for Python," 2014. Available:
> <https://github.com/fmfn/BayesianOptimization>

and, it deployed the model on he AERPAW testbed:

> V. Marojevic, I. Guvenc, R. Dutta, M. Sichitiu, and B. Floyd,
> "Advanced Wireless for Unmanned Aerial Systems:5G Standardization,
> Research Challenges, and AERPAW Experimentation Platform", IEEE Vehic.
> Technol. Mag., vol. 15, no. 2. pp. 22-30, June 2020. DOI:
> 10.1109/MVT.2020.2979494.


------------------------------------------------------------------------

####  📝 Specific requirements:

-   For full credit, you should achieve 10m or less estimation error by
    the end of the five-minute flight.
-   and, your fitted model should not show signs of severe overfitting
    or under-modeling - it should show a reasonable approximation of the
    function of signal strength vs position over the search space.

------------------------------------------------------------------------

## Framing the problem
 
We are going to estimate the hikers' position based on the premise that
the received signal strength is highest when the UAV is at the same
latitude and longitude as the hikers.

We will frame our machine learning problem as follows:

-   features X: latitude, longitude
-   target variable y: received signal strength

In other words, given a coordinate (latitude and longitude) we want to
predict the received signal strength at that location.

You can learn more about the problem, and our approach to solving it with 
a Gaussian Processing Regression, at: <a target="_blank" href="https://colab.research.google.com/github/teaching-on-testbeds/uav-wireless-localization/blob/main/kernel_find_a_rover_synthetic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

However, we don't care as much if our model is bad at predicting the
signal strength in places where it is low! Our *true* goal is to predict
where the target variable will be highest. We will decide how "good" our
model is by computing the mean squared error of the position estimate:
the distance between the true location of the hikers, and the coordinate
that our model predicts has the highest received signal strength.
 
## Rover search experiment on AERPAW
 
This sequence assumed you have already

-   [created an account on AERPAW and joined a project](https://teaching-on-testbeds.github.io/hello-aerpaw/index#create-an-account-on-aerpaw) (one-time step)
-   [created an experiment with a UGV and UAV and initiated development mode](https://teaching-on-testbeds.github.io/hello-aerpaw/index#start-an-experiment) (one-time step until you retire the experiment)

Finally, when you are ready to test your model in the "real" search
environment, you need to [set up access to experiment resources](https://teaching-on-testbeds.github.io/hello-aerpaw/index#access-experiment-resources),
including:

-   connecting your computer to the AERPAW VPN,
-   opening an SSH session to the experiment console,
-   opening an SSH session to the UAV VM (node 1 in the experiment),
-   opening an SSH session to the UGV VM (node 2 in the experiment).
-   if you will use QGroundControl: connecting QGroundControl, and setting up the `AFAR Rover.kml` geofence,

You may review [Hello, AERPAW](https://teaching-on-testbeds.github.io/hello-aerpaw/) as a
reference for those last steps.
 
## Set up experiment on UAV and UGV VMs

Now, we will configure applications that will run in the experiment -
the radio transmitter (on UGV) and radio receiver (on UAV), and the
Bayes search on the UAV.

Inside the SSH session on the UAV (node 1 in the experiment), install
the `bayesian-optimization` package, which we will use to implement a
Bayes search:

```bash
# run on E-VM-M1 (UAV node)
python3 -m pip install --target=/root/Profiles/vehicle_control/RoverSearch bayesian-optimization==2.0.0 numpy==1.26.4 scikit_learn==1.5.2
```

Download the `rover-search.py` script:

```bash
# run on E-VM-M1 (UAV node)
wget https://raw.githubusercontent.com/teaching-on-testbeds/uav-wireless-localization/refs/heads/main/rover_search.py -O  /root/Profiles/vehicle_control/RoverSearch/rover_search.py
```

and the signal power plotting script:

```bash
# run on E-VM-M1 (UAV node)
wget https://raw.githubusercontent.com/teaching-on-testbeds/hello-aerpaw/refs/heads/main/resources/plot_signal_power.py -O  /root/plot_signal_power.py
```

Still in the SSH session on the UAV (node 1 in the experiment), set up
the applications that will run during our experiment - a radio receiver
and a vehicle control script that implements our search with Gaussian
Process Regression and Bayesian Optimization:

```bash
# run on E-VM-M1 (UAV node)
cd /root/Profiles/ProfileScripts/Radio 
cp Samples/startGNURadio-ChannelSounder-RX.sh startRadio.sh 

cd /root/Profiles/ProfileScripts/Vehicle
cp Samples/startRoverSearch.sh startVehicle.sh

cd /root
```

We will also change one parameter of the radio receiver. Run:

```bash
# run on E-VM-M1 (UAV node)
sed -i 's/^SPS=.*/SPS=8/' "/root/Profiles/ProfileScripts/Radio/Helpers/startchannelsounderRXGRC.sh"
```

 
Then, open the experiment script for editing

```bash
# run on E-VM-M1 (UAV node)
cd /root
nano /root/startexperiment.sh
```

and at the bottom of this file, remove the `#` comment sign (if there is one) 
next to `./Radio/startRadio.sh` and `./Vehicle/startVehicle.sh`, so that the 
end of the file looks like this:

```
./Radio/startRadio.sh
#./Traffic/startTraffic.sh
./Vehicle/startVehicle.sh
```

Hit Ctrl+O and then hit Enter to save the file. Then use Ctrl+X to exit
and return to the terminal.

 
Now we will set up the UGV.

Inside an SSH session on the UGV (node 2 in the experiment), set up the
applications that will run during our experiment - a radio transmitter
and a vehicle GPS position logger:

```bash
# run on E-VM-M2 (UGV node)
cd /root/Profiles/ProfileScripts/Radio 
cp Samples/startGNURadio-ChannelSounder-TX.sh startRadio.sh 

cd /root/Profiles/ProfileScripts/Vehicle
cp Samples/startGPSLogger.sh startVehicle.sh

cd /root
```

We will also change one parameter of the radio transmitter. Run:

```bash
# run on E-VM-M2 (UGV node)
sed -i 's/^SPS=.*/SPS=8/' "/root/Profiles/ProfileScripts/Radio/Helpers/startchannelsounderTXGRC.sh"
```

Then, open the experiment script for editing

```bash
# run on E-VM-M2 (UGV node)
cd /root
nano /root/startexperiment.sh
```

and at the bottom of this file, remove the `#` comment sign (if there is one) 
next to `./Radio/startRadio.sh` and `./Vehicle/startVehicle.sh`, so that the 
end of the file looks like this:

```
./Radio/startRadio.sh
#./Traffic/startTraffic.sh
./Vehicle/startVehicle.sh
```

Hit Ctrl+O and then hit Enter to save the file. Then use Ctrl+X to exit
and return to the terminal.

 
## Set up steps in experiment console


> **Note**: a video of this section is included at the end of the section.
 
On the experiment console, run

```bash
# run on OEO-CONSOLE VM
./startOEOConsole.sh
```

and add a column showing the position of each vehicle; in the experiment
console run

```bash
# run on OEO-CONSOLE VM, inside the experiment console process
add vehicle/position
```

and you will see a `vehicle/position` column added to the end of the
table.

Then, in this experiment console window, set the start position of the
UGV (node 2):

```bash
# run on OEO-CONSOLE VM, inside the experiment console process
2 start_location 35.729 -78.699
```

and restart the controller on the UGV, so that the change of start
location will take effect:

```bash
# run on OEO-CONSOLE VM, inside the experiment console process
2 restart_cvm
```
 
If you are also watching in QGroundControl: In QGroundControl, the
connection to the UGV may be briefly lost. Then it will return, and the
UGV will be at the desired start location.
 
Even if you are not watching in QGroundControl, you will see in the
`vehicle/position` column in the experiment console that the UGV (node
2) is at the position we have set.


<video width="1280" height="720" controls autoplay muted loop>
<source src="https://teaching-on-testbeds.github.io/uav-wireless-localization/video/aerpaw_exp_console_an.mp4" type="video/mp4">
 Your browser does not support the video tag.
</video>
 
## Rover search experiment with default position and default model

Now we are ready to run an experiment!
 
### Reset
 
> **Note**: a video of this section is included at the end of the section.

 
Start from a "clean slate" - on the UAV VM (node 1) and the UGV VM (node
2), run

```bash
# run on E-VM-M1 (UAV node) and ALSO on E-VM-M2 (UGV node)
cd /root
./stopexperiment.sh
```

to stop any sessions that may be lingering from previous experiments.
 
You should also reset the virtual channel emulator in between runs - on
*either* VM (node 1 or node 2) run

```bash
# run on E-VM-M1 (UAV node) OR on E-VM-M2 (UGV node)
./reset.sh
```


<video width="1280" height="720" controls autoplay muted loop>
<source src="https://teaching-on-testbeds.github.io/uav-wireless-localization/video/aerpaw_reset_experiment_an.mp4" type="video/mp4">
 Your browser does not support the video tag.
</video>
 
 
### Start experiment

 
> **Note**: a video of this section is included at the end of the section.

 
On the UGV VM (node 2), run

```bash
# run on E-VM-M2 (UGV node)
cd /root
./startexperiment.sh
```
 
In the terminal in which you are connected to the experiment console
(with a table showing the state of the two vehicles) run

```bash
# run on OEO-CONSOLE VM, inside the experiment console process
2 arm
```

In this table, for vehicle 2, you should see a "vehicle" and "txGRC"
entry in the "screens" column.

 
On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
cd /root
./startexperiment.sh
```

and wait a few moments, until you see the new processes appear in the
"screens" column of the experiment console.
 
Then check the log of the vehicle navigation process by running (on the
UAV VM, node 1):

```bash
# run on E-VM-M1 (UAV node)
tail -f Results/$(ls -tr Results/ | grep vehicle_log | tail -n 1 )
```

You should see a message

    Guided command attempted. Waiting for safety pilot to arm

When you see this message, you can use Ctrl+C to stop watching the
vehicle log.

 
In the experiment console, run

```bash
# run on OEO-CONSOLE VM, inside the experiment console process
1 arm
```

to arm this vehicle. It will take off, reach altitude 50, and begin to
search for the UGV. 

You can monitor the position of the UAV by watching the flight in
QGroundControl, or you can watch the position in the experiment console.

While the search is ongoing, monitor the received signal power by
running (on the UAV VM, node 1):

```bash
# run on E-VM-M1 (UAV node)
python3 plot_signal_power.py
```

and confirm that you see a stream of radio measurements, and that the
signal is stronger when the UAV is close to the UGV. 
 
The experiment will run for 5 minutes from the time that the UAV reaches
altitude. Then, the UAV will return to its original position and land.

 
When you see that the "screens" column in the experiment console no
longer includes a "vehicle" entry for the UAV (node 1), its "mode" is
LAND, and its altitude is very close to zero, then you know that the
experiment is complete. You must wait for the experiment to completely
finish, because the data files are only written at the end of the
experiment.


<video width="1280" height="720" controls autoplay muted loop>
<source src="https://teaching-on-testbeds.github.io/uav-wireless-localization/video/aerpaw_start_experiment_an.mp4" type="video/mp4">
 Your browser does not support the video tag.
</video>
 
### Transfer data from AERPAW for analysis

 
Once your experiment is complete, you can transfer a CSV file of the
search progress and the final optimizer state from AERPAW to your own
laptop, for further analysis.

 
On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep ROVER_SEARCH | tail -n 1 )
```

to get the name of the CSV file.

On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep opt_final | tail -n 1 )
```

to get the name of the "pickled" optimizer file.
 
Then, in a *local* terminal (**not** inside any SSH session), `cd` to a directory where you have write access (if necessary) and run

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/ROVER_SEARCH_DATA_XXXXXX.csv ROVER_SEARCH_DATA_default.csv
```

where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the rover search CSV filename you identified above

Also run 

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/opt_final_XXXXXX.pickle opt_final_default.pickle
```


where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the "pickled" optimizer filename you identified above

You may be prompted for the passphrase for your key, if you set a
passphrase when generating the key.

 
After you run these `scp` commands, you should have a `ROVER_SEARCH_DATA_default.csv` file and an
`opt_final_default.pickle` file on your laptop. 
 
### Analyze experiment results

To analyze the experiment results, open the following Colab notebook: <a target="_blank" href="https://colab.research.google.com/github/teaching-on-testbeds/uav-wireless-localization/blob/main/kernel_find_a_rover_aerpaw.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Execute the first few cells in that notebook, to import libraries and functiosn.

Then, use the file browser in Google Colab to upload the `ROVER_SEARCH_DATA_default.csv` file and
`opt_final_default.pickle` file to Colab. 

Run the "Analyze experiment results from "default" experiment" section to visualize the search.


 
## Rover search with new location and default model

Next, you will re-run the experiment, but with the rover at a different location. Use the cell at 
the beginning of the "Analyze results from rover search with new location" section of the Colab notebook
to generate a "personal" new start position for the rover.

Now, you will re-do the "Rover search experiment" section. You
will:

-   Repeat "Set up steps in experiment console", but use the "personal" latitude 
    and longitude generated in the Colab notebook for *your* net ID.
-   Repeat the "Run experiment" steps (including "Reset" and "Start
    experiment").
    
Once your experiment is complete, you will transfer the CSV file of the
search progress and the final optimizer state from AERPAW to your own
laptop, for further analysis.
 
On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep ROVER_SEARCH | tail -n 1 )
```

to get the name of the new CSV file.

On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep opt_final | tail -n 1 )
```

to get the name of the new "pickled" optimizer file.
 
Then, in a *local* terminal (**not** inside any SSH session), `cd` to a directory where you have write access (if necessary) and run

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/ROVER_SEARCH_DATA_XXXXXX.csv ROVER_SEARCH_DATA_new.csv
```

where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the rover search CSV filename you identified above

Also run 

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/opt_final_XXXXXX.pickle opt_final_new.pickle
```


where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the "pickled" optimizer filename you identified above

You may be prompted for the passphrase for your key, if you set a
passphrase when generating the key.

 
After you run these `scp` commands, you should have a `ROVER_SEARCH_DATA_new.csv` file and an
`opt_final_new.pickle` file on your laptop. These are the results for the default model, but 
with the rover at the **new** location.

Upload these files to the Colab notebook. 

Then, in the rest of the "Analyze results from rover search with new location" section
you will repeat the analysis for your new experiment (with the
"hikers' position" at this new location).


## Rover search with new location AND customized model

Finally, you will re-run the experiment, but you may modify the kernel function and/or the 
utility function of the Bayesian optimization, in order to satisfy the specific requirements:

---

📝 Specific requirements:

-   For full credit, you should achieve 10m or less estimation error by
    the end of the five-minute flight.
-   and, your fitted model should not show signs of severe overfitting
    or under-modeling - it should show a reasonable approximation of the
    function of signal strength vs position over the search space.

---

Currently, the optimizer is configured as:

```python
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
```

but, you know these are not the ideal settings for finding the lost hikers. You can modify this - specifically, you can:

* set the `kappa` argument of the utility function, 
* add a `WhiteKernel()`, 
* and/or set the bounds of the kernel hyperparameters.

(you don't *have* to do all of these, just do what you believe will be effective based on your previous experiments).


Edit the rover search script:

```bash
# run on E-VM-M1 (UAV node)
nano /root/Profiles/vehicle_control/RoverSearch/rover_search.py
```

scroll to [the part where the utility function, optimizer, and kernel are defined](https://github.com/teaching-on-testbeds/uav-wireless-localization/blob/main/rover_search.py#L58), and edit them.

Then use Ctrl+O and Enter to save the file, and Ctrl+X to exit.

Now, you will re-do the "Rover search experiment" section. You
will:

-   Repeat "Set up steps in experiment console", but use the "personal" latitude 
    and longitude generated in the Colab notebook for *your* net ID.
-   Repeat the "Run experiment" steps (including "Reset" and "Start
    experiment").
    
Once your experiment is complete, you will transfer the CSV file of the
search progress and the final optimizer state from AERPAW to your own
laptop, for further analysis.
 
 
On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep ROVER_SEARCH | tail -n 1 )
```

to get the name of the new CSV file.

On the UAV VM (node 1), run

```bash
# run on E-VM-M1 (UAV node)
echo /root/Results/$(ls -tr Results/ | grep opt_final | tail -n 1 )
```

to get the name of the new "pickled" optimizer file.
 
Then, in a *local* terminal (**not** inside any SSH session), `cd` to a directory where you have write access (if necessary) and run

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/ROVER_SEARCH_DATA_XXXXXX.csv ROVER_SEARCH_DATA_custom.csv
```

where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the rover search CSV filename you identified above

Also run 

```bash
# run on your local terminal -NOT inside an SSH session
scp -i ~/.ssh/id_rsa_aerpaw root@192.168.X.1:/root/Results/opt_final_XXXXXX.pickle opt_final_custom.pickle
```


where

-   in place of the address with the `X`, you use the address you
    identified in the manifest,
-   in place of the file name with the `XXXXXX` in the filename, 
    you substitute the "pickled" optimizer filename you identified above

You may be prompted for the passphrase for your key, if you set a
passphrase when generating the key.

 
After you run these `scp` commands, you should have a `ROVER_SEARCH_DATA_custom.csv` file and an
`opt_final_custom.pickle` file on your laptop. These are the results for the **custom** model, 
with the rover at the **new** location.

Upload these files to the Colab notebook. 

Then, in the "Analyze results from rover search with custom model" section
you will repeat the analysis for your new experiment (with the
"hikers' position" at this new location, and using your custom model).


Verify that you have met the specific requirements. Then, comment on the results, specifically:

* what changes did you make do the default settings of the optimizer and model?
* how has the appearance of the fitted model changed from the previous experiment, and why?
* what change do you see in the fitted model kernel parameters? 

