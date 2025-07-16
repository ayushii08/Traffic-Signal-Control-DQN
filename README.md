# Traffic Signal Control Adaptation using Deep Q-Learning with Experience Replay
This project presents a **reinforcement learning-based adaptive traffic signal control** system that leverages **Deep Q-Networks (DQN)** with experience replay to minimize vehicular congestion at a single urban intersection, simulated using **SUMO (Simulation of Urban MObility)**.

Urban intersections suffer from inefficient traffic signal phasing due to static control policies. **Reinforcement Learning (RL)** provides a promising alternative by learning dynamic signal control strategies based on real-time traffic states.

This project applies **Deep Q-Learning (DQL)** to this problem domain, wherein the agent learns to:
- Observe traffic conditions
- Take phase-switching actions
- Receive environment feedback (rewards)
- Improve its policy via trial and error

We simulate traffic patterns using **SUMO**, allowing for flexible environment design and high-fidelity vehicle behavior modeling.
## Methodology

* ### State Representation
The intersection is represented as a grid of cells. Each state vector encodes the **occupancy status** of each grid segment (0 or 1), forming a **fixed-length state vector** (e.g., 80 binary inputs).

* ### Action Space
There are 4 discrete traffic light phase actions, each corresponding to a specific signal direction (e.g., NS-Green, EW-Green).

* ### Reward Function
The reward is the **negative cumulative waiting time** of vehicles, encouraging the agent to **minimize delay**.

```python
reward = -(total_wait_time)
```




## Environment Setup 
This project is built for simulation-driven deep reinforcement learning using the SUMO (Simulation of Urban Mobility) engine in combination with Python libraries like TensorFlow and Traci. It has been validated both in local environments (macOS, Ubuntu) and on headless platforms like Google Colab (with GUI disabled).

*For best performance, GPU acceleration is recommended, though this project has been optimized for fast training even without it by reducing training complexity.*
---

### System Requirements

- **OS:** macOS (Apple Silicon M1), Ubuntu (20.04+ recommended), or Google Colab
- **Python:** 3.8 or above
- **RAM:** ≥ 8 GB
- **SUMO version:** 1.14.0+ (or stable)
- **TensorFlow version:** ≥ 2.9 (with or without GPU)

---

### Required Dependencies

| Tool / Library     | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `sumo`, `sumo-tools` | Open-source microscopic traffic simulator                                 |
| `traci`            | Traffic control interface to manipulate SUMO during runtime                |
| `sumolib`          | Utility library for interacting with SUMO's network and route definitions  |
| `tensorflow`       | Deep learning backend for Q-network                                         |
| `matplotlib`       | Visualization library for training/testing metrics                         |
| `graphviz`, `pydot`| Required for SUMO graph/network manipulation                                |

---
### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/ayushii08/Traffic-Signal-Control-DQN.git
cd Traffic-Signal-Control-DQN
```
#### 2. Install SUMO
* macOS (via Homebrew):
```
brew install sumo
```
* Ubuntu (via APT):
```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
*After installation, verify SUMO tools:*
```
sumo --version
```
#### 3. Configure Environment Variable
SUMO uses the environment variable SUMO_HOME to locate its tools directory.
* Add this to your .bashrc or .zshrc:
  ```
  export SUMO_HOME="/usr/share/sumo"  # or wherever SUMO is installed
  export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
  ```
* Reload the shell:
  ```
  source ~/.bashrc  # or source ~/.zshrc
  ```
  macOS Homebrew path may look like:
  ```
  /opt/homebrew/Cellar/sumo/<version>/share/sumo/
  ```
####  4. Install Python Dependencies
```
pip install tensorflow traci sumolib matplotlib pydot
sudo apt install graphviz  # Only needed on Linux for rendering traffic routes
```
If you're using Anaconda:
```
conda create -n traffic-dqn python=3.8
conda activate traffic-dqn
pip install -r requirements.txt  # Optional if you’ve created one
```
#### 5. Running on Google Colab (Headless Environment)
(GUI-based SUMO simulations are not supported in Colab)
* Set gui = False in Training_Setup.ini and Testing_Setup.ini
* Upload the project files to your working directory
* Use the %cd magic command to navigate into the project root
* Run training or testing using:
  ```
  !python Train_Main.py
  !python Test_Main.py
  ```
* Output files like reward plots and queue data are saved under /models/model_x/test/

---

#### Model Architecture

The Deep Q-Network (DQN) used in this project approximates the action-value function \( Q(s, a) \), which predicts the cumulative reward expected after taking action \( a \) in state \( s \).

### Network Design

- **Input Layer:** Receives a binary vector representing the intersection state (e.g., 80 bits for 80 cells).
- **Hidden Layers:** 2 fully connected layers (configurable width via `layerWidth`).
- **Activation Function:** ReLU
- **Output Layer:** 4 neurons representing Q-values for each action (traffic signal phase)

###  Architecture Parameters

| Parameter     | Description                                | Value (default) |
|---------------|--------------------------------------------|-----------------|
| `numStates`   | Input state size                           | 80              |
| `numActions`  | Number of discrete actions (phases)        | 4               |
| `layerWidth`  | Number of neurons in each hidden layer     | 64              |
| `numLayers`   | Total hidden layers                        | 2               |
| `learningRate`| Adam optimizer learning rate               | 0.001           |
| `gamma`       | Discount factor for future rewards         | 0.75            |

> The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss.

---

## 3. Training Configuration

All training hyperparameters are defined in `Training_Setup.ini`.

### Key Parameters

| Parameter          | Description                                           | Example Value |
|-------------------|-------------------------------------------------------|---------------|
| `numEpisodes`      | Number of training episodes                          | 30            |
| `maxSteps`         | Steps per episode                                    | 7200          |
| `numCars`          | Cars per episode                                     | 100           |
| `batchSize`        | Experience replay mini-batch size                    | 16            |
| `trainingEpochs`   | Training epochs after each episode                   | 3             |
| `greenDuration`    | Time (s) for each green phase                        | 10            |
| `yellowDuration`   | Yellow light duration (s)                            | 4             |
| `minMemorySize`    | Minimum samples before training begins               | 500           |
| `maxMemorySize`    | Max replay memory capacity                           | 5000          |
| `gamma`            | Discount factor for Bellman update                   | 0.75          |

### Training Flow

1. Initialize SUMO environment.
2. At each step:
   - Observe state.
   - Select action via epsilon-greedy policy.
   - Execute action in SUMO.
   - Store experience tuple.
   - Sample and train on mini-batches from memory.
3. Save model weights and plots at the end of each episode.

```bash
python Train_Main.py
```
---

## 4. Testing Protocol

The trained model is evaluated in a separate testing phase using parameters defined in `Testing_Setup.ini`. The environment generates traffic with a distinct seed and higher vehicle density to assess model generalization.

### ⚙️ Testing Parameters

| Parameter         | Description                                        | Example Value |
|------------------|----------------------------------------------------|---------------|
| `gui`             | Enable/disable SUMO GUI                           | `False`        |
| `maxSteps`        | Total simulation steps per episode                | `7200`        |
| `numCars`         | Number of vehicles injected                       | `1000`        |
| `episodeSeed`     | Random seed for test episode                      | `10000`       |
| `greenDuration`   | Fixed duration for green phase                    | `10` seconds  |
| `yellowDuration`  | Fixed yellow light time                           | `4` seconds   |
| `modelForTesting` | Trained model version to load                     | `2`           |

###  Running Tests

Run the testing script to simulate traffic behavior with the selected trained model:

```bash
python Test_Main.py
```
---

## 5. Results Visualization

After training and testing, the framework automatically generates visualizations to help evaluate the model's performance.

### Reward Plot (`plot_reward.png`)

- **Purpose**: Tracks the total reward collected during the simulation.
- **Insight**: Higher rewards (less negative) indicate better control policies and smoother traffic flow.
- **Source**: Calculated as the negative sum of cumulative wait time at the intersection.

📍 *Path:* `models/model_<id>/test/plot_reward.png`

---

###  Queue Length Plot (`plot_queue.png`)

- **Purpose**: Displays the evolution of average vehicle queue length during the episode.
- **Insight**: Lower average queue lengths over time reflect more efficient signal timing.
- **Goal**: Minimize traffic congestion and stabilize flow at intersections.

📍 *Path:* `models/model_<id>/test/plot_queue.png`

---

##  6. File Structure

The repository is organized as follows:

```bash
Traffic-Signal-Control-DQN/
│
├── Train_Main.py               # Entry point for model training
├── Test_Main.py                # Entry point for model evaluation
├── Model.py                    # DQN model architecture for training and testing
├── Helpers.py                  # Utilities: route generation, plotting, logging
├── Train_Simulation.py         # Handles SUMO interactions during training
├── Test_Simulation.py          # Handles SUMO interactions during testing
│
├── Training_Setup.ini          # Training hyperparameters and simulation config
├── Testing_Setup.ini           # Testing hyperparameters and simulation config
│
├── Sumo_environment/           # SUMO simulation configuration
│   ├── environment.net.xml     # Road network file
│   └── sumo_config.sumocfg     # Connects network and traffic route files
│
├── models/                     # Directory to save trained models and outputs
│   └── model_<id>/             # Versioned model directories
│       ├── model.h5            # Saved DQN model weights
│       └── test/               # Test logs, reward/queue plots, setup configs
│
├── README.md                   # Project overview and documentation
└── .gitignore                  # Ignore configuration for Git version control
```

  


## Acknowledgement
[Deep Q-Learning Agent for Traffic Signal Control1](https://github.com/Reinforcement-Learning-F22/Traffic-Signal-Control-using-Deep-Q-Learning)

[Deep Q-Learning Agent for Traffic Signal Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control)

## References
[Tensorflow GPU Installation](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)


    
    
    
