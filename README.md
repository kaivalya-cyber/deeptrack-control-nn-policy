# Car Racing Simulation (REINFORCE)

It trains a car to race around a track using **REINFORCE** and Gymnasium’s **CarRacing-v3** (Box2D) environment.

- **Environment**: Top-down car, continuous actions (steer, gas, brake), 96×96 RGB image observation.
- **Policy**: CNN that takes the image and outputs a Gaussian policy over 3 actions.
- **Training**: Same REINFORCE setup as the Hopper track: multi-seed, episode returns, learning curve plot.

Use **python3** and **pip3** for all commands.

## Setup

From this directory (`car_racing/`):

```bash
pip3 install -r requirements.txt
```

## Run

```bash
python3 train_reinforce_car.py
```

**Options**

- `--episodes N` — Episodes per seed (default: 2000).
- `--seeds N` — Number of seeds (default: 3).
- `--no-show` — Only save the plot; do not open a window.
- `--render` — Show the car racing window during training (opens a pygame window; requires a display).

**Examples**

- Quick run with viewer: `python3 train_reinforce_car.py --episodes 200 --seeds 1 --render`
- Headless: `python3 train_reinforce_car.py --episodes 500 --seeds 2 --no-show`

The learning curve is saved as **learning_curve_car.png** in this folder.

**Rendering:** With `--render`, a CarRacing game window opens and updates each step. If you see a “video device not available” or similar error (e.g. on a headless server), run without `--render`; rendering requires a display.

## Repository Structure and Logic

### File Structure

- **`train_reinforce_car.py`**: The main entry point. It contains the model definition, the RL agent logic, and the training loop.
- **`requirements.txt`**: Lists the Python dependencies (Gymnasium, PyTorch, etc.).
- **`learning_curve_car.png`**: An example output plot showing agent performance over episodes.

### Logic and Architecture

#### CNN Policy (`CNNPolicy`)
The agent uses a Convolutional Neural Network to process the 96x96x3 image observations from the environment.
- **Feature Extractor**: Three convolutional layers with Tanh activations.
- **Policy Heads**: Two linear layers that output the **mean** and **standard deviation** for each of the 3 actions (steer, gas, brake).
- **Action Sampling**: Actions are sampled from a Normal distribution defined by these means and standard deviations.

#### REINFORCE Algorithm
The training follows the REINFORCE (Monte Carlo Policy Gradient) logic:
1. **Sampling**: For each step in an episode, the agent samples an action and stores its log-probability.
2. **Returns**: At the end of the episode, it calculates the discounted cumulative rewards (returns) for each step.
3. **Update**: The policy is updated by minimizing the loss: `loss = -sum(log_prob * return)`. This increases the probability of actions that led to higher rewards.
