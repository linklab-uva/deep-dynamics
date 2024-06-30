# Deep Dynamics

Deep Dynamics is a physics-constrained neural network (PCNN) designed to model the complex dynamics observed in a high-speed, competitive racing environment. Using a historical horizon of the vehicle's state and control inputs, Deep Dynamics learns to produce accurate coefficient estimates for a dynamic single-track model that best describes the vehicle's motion. Specifically, this includes Pacejka Magic Formula coefficients for the front and rear wheels, coefficients for a linear drivetrain model, and the vehicle's moment of inertia. The Physics Guard layer ensures that estimated coefficients always lie within their physically-meaningful range, determined by the meaning behind each coefficient.

## Installation

It is recommended to create a new conda environment:

```
conda create --name deep_dynamics python=3.10
conda activate deep_dynamics
```

To install Deep Dynamics:

```
git clone git@github.com:linklab-uva/deep-dynamics.git
cd deep-dynamics/
pip install -e .
```

## Processing Data

Data collected from the [Bayesrace vehicle dynamics simulator](https://github.com/jainachin/bayesrace) and the AV-21 full-scale autonomous racecar competing in the [Indy Autonomous Challenge](https://www.indyautonomouschallenge.com/) run by the [Cavalier Autonomous Racing Team](https://autonomousracing.dev/) is provided for training and testing.

Bayesrace:
```
1. deep_dynamics/data/DYN-PP-ETHZ.npz
2. deep_dynamics/data/DYN-PP-ETHZMobil.npz
```

Indy Autonomous Challenge (IAC):
```
1. deep_dynamics/data/LVMS_23_01_04_A.csv
2. deep_dynamics/data/LVMS_23_01_04_B.csv
3. deep_dynamics/data/Putnam_park2023_run2_1.csv
4. deep_dynamics/data/Putnam_park2023_run4_1.csv
5. deep_dynamics/data/Putnam_park2023_run4_2.csv
```

To convert data from Bayesrace to the format needed for training:

```
cd tools/
./bayesrace_parser.py {path to dataset} {historical horizon size}
```

Historical horizon size refers to the number of historical state and control input pairs used as features during training. The process is similar for IAC data:

```
cd tools/
./csv_parser.py {path to dataset} {historical horizon size}
```

The resulting file will be stored under `{path to dataset}_{historical_horizon_size}.npz`.

## Model Configuration

Configurations for Deep Dynamics and the closely related [Deep Pacejka Model](https://arxiv.org/pdf/2207.07920.pdf) are provided under `deep_dynamics/cfgs/`. The items listed under `PARAMETERS` are the variables estimated by each model. The ground-truth coefficient values used for the Bayesrace simulator are displayed next to each coefficient (i.e. `Bf: 5.579` indicates the coefficient Bf was set to 5.579). The ground-truth values are only used for evaluation purposes, they are not accessible to the models during training. The Physics Guard layer requires ranges for the coefficients estimated by Deep Dynamics and can be specified with the `Min` and `Max` arguments.

Certain properties for the vehicle are required for training. Provided under `VEHICLE_SPECS`, this includes the vehicle's mass and the distance from the vehicle's center of gravity (COG) to the front and rear axles (`lf` and `lr`). The Deep Pacejka Model also requires the vehicle's moment of inertia (`Iz`) is specified.

Under `MODEL`, the layers for each model can be specified. The `HORIZON` refers to the historical horizon of state and control inputs used as features. Under `LAYERS`, the input and hidden layers of the model can be specified. Lastly, the optimization parameters are provided under `OPTIMIZATION`.

## Model Training

To run an individual training experiment, use:

```
cd deep_dynamics/model/
python3 train.py {path to cfg} {path to dataset} {name of experiment}
```

The optional flag `--log_wandb` can also be added to track results using the [Weights & Biases Platform](https://wandb.ai/site). Model weights will be stored under `../output/{name of experiment}` whenever the validation loss decreases below the previous minima.

To run multiple trials in parallel for hyperparameter tuning, use:

```
cd deep_dynamics/model/
python3 tune_hyperparameters.py {path to cfg}
```

The desired dataset must be manually specified within `tune_hyperparameters.py` as well as the ranges for the hyperparameter tuning experiment. Trials are run using the [RayTune scheduler](https://docs.ray.io/en/latest/tune/index.html).

## Model Evaluation

To evaluate an individual model, use:

```
cd deep_dynamics/model/
python3 evaluate.py {path to cfg} {path to dataset} {path to model weights}
```

This will evaluate the model's RMSE and maximum error for the predicted state variables across the specified dataset. Additionally, the optional flag `--eval_coeffs` can be used to compare the mean and standard deviation of the model's internally estimated coefficients.

To evaluate multiple trials from hyperparameter tuning, use:

```
cd deep_dynamics/model/
python3 test_hyperparameters.py {path to cfg}
```

You can cite this work using:

```
@ARTICLE{10499707,
  author={Chrosniak, John and Ning, Jingyun and Behl, Madhur},
  journal={IEEE Robotics and Automation Letters}, 
  title={Deep Dynamics: Vehicle Dynamics Modeling With a Physics-Constrained Neural Network for Autonomous Racing}, 
  year={2024},
  volume={9},
  number={6},
  pages={5292-5297},
  keywords={Vehicle dynamics;Mathematical models;Tires;Solid modeling;Predictive models;Physics;Autonomous vehicles;Deep learning methods;model learning for control;dynamics},
  doi={10.1109/LRA.2024.3388847}}
```

You can also read my Master's thesis on this work [here](https://libraetd.lib.virginia.edu/public_view/qr46r2095) :)
