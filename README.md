# CarRacing-v3 DQN Agent

This project implements a Deep Q-Network (DQN) agent to play the CarRacing-v3 game from OpenAI Gymnasium. The agent learns to drive using visual input (images) and reinforcement learning techniques.

---

## Features

- **Deep Q-Network (DQN)** with convolutional layers for image-based state processing.
- **Experience Replay** and **Target Network** for stable training.
- **Epsilon-Greedy Policy** for exploration/exploitation balance.
- **Training Progress Visualization** and **Model Checkpointing**.
- **Evaluation and Video Generation** of the trained agent.

---

## Setup Instructions

1. **Clone or Download the Repository**
   - Place all project files in a working directory.

2. **Install Required Packages**
   - This project requires Python 3.8+.
   - Install dependencies using pip:
     ```bash
     pip install gymnasium[box2d] torch numpy matplotlib imageio box2d-py
     ```

3. **(Optional) Google Colab**
   - The code is compatible with Google Colab. If running in Colab, ensure you install the following:
     ```python
     !pip install swig
     !pip install box2d-py
     !pip install stable-baselines3[extra] gym
     ```

---

## Running the Project

### 1. Training the Agent

- Run the main training script (e.g., `CarRacing_Agent.ipynb` or the provided Python script).
- The agent will train for a specified number of episodes, periodically saving model checkpoints (e.g., `car_racing_dqn_episode_200.pth`).
- Training progress (rewards per episode) will be plotted and saved as `car_racing_training_rewards.png`.

### 2. Evaluating the Agent

- After training, load a saved model checkpoint.
- Run the evaluation script to let the agent play the game using a greedy policy (no exploration).
- The evaluation will capture frames and save a video (e.g., `evaluation.mp4`) of the agent’s performance.

### 3. Saving and Loading Models

- The agent’s state (networks, optimizer, epsilon) is saved in `.pth` files.
- Use the `agent.save(path)` and `agent.load(path)` methods to save/load models.

---

## File Structure

- `CarRacing_Agent.ipynb` / `.py` : Main code for training and evaluation.
- `car_racing_dqn_episode_200.pth` : Model checkpoints saved during training.
- `car_racing_training_rewards.png` : Training reward plot.
- `Model200.mp4` : Video of the agent’s performance.
- `README.md` : This file.

---

## Notes

- The environment uses a **discrete action space** for simplicity.
- Preprocessing converts RGB images to grayscale and normalizes them for the neural network.
- Training can be time-consuming; using a GPU is recommended.
- For best results, adjust hyperparameters (learning rate, batch size, etc.) as needed.

---

## Credits

- Developed by Mir Maiti, Ishaan Bahl, and Chandan Jyoti Das.
- Based on OpenAI Gymnasium’s CarRacing-v3 environment.

---

## Contact

For questions or suggestions, please contact the project contributors.

---
