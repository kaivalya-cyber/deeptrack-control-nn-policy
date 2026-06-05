# Car Racing Simulation (Actor-Critic)

It trains a car to race around a track using an **Actor-Critic** agent and Gymnasium’s **CarRacing-v3** (Box2D) environment.

- **Environment**: Top-down car, continuous actions (steer, gas, brake), 96×96 image observation.
- **Preprocessing**: Observation is converted to grayscale and stacked (4 frames) to provide temporal information.
- **Policy**: CNN that takes the 4-channel image stack and outputs a Gaussian policy over 3 actions plus a state-value estimate (Critic).
- **Training**: Actor-Critic approach with advantage calculation and entropy regularization.

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
- `--lr LR` — Learning rate (default: 1e-4).
- `--entropy E` — Entropy coefficient for regularization (default: 0.01).
- `--critic C` — Weight for the critic loss (default: 0.5).
- `--load PATH` — Load a pre-trained model from the specified path.

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
The agent uses a Convolutional Neural Network to process the 4-channel grayscale image stacks (96x96x4).
- **Feature Extractor**: Three convolutional layers with Tanh activations.
- **Policy Heads**: Two linear layers that output the **mean** and **standard deviation** for each of the 3 actions (steer, gas, brake).
- **Value Head**: A linear layer that outputs the **state value** (Critic).
- **Action Sampling**: Actions are sampled from a Normal distribution defined by these means and standard deviations.

#### Actor-Critic Logic
The training uses an Actor-Critic approach with the following components:
1. **Advantages**: Calculated as `Return - Value(State)`. This reduces variance compared to using raw returns.
2. **Actor Loss**: Minimized using `-log_prob * advantage`.
3. **Critic Loss**: MSE between the predicted value and the actual returns.
4. **Entropy Regularization**: Encourages exploration by adding an entropy term to the loss.
5. **Model Saving**: The best model (highest average return) for each seed is automatically saved as `best_model_seed_N.pt`.
