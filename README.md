
# Drone-Zone: Autonomous Flight using Q-Learning

🚀 Drone-Zone is a Reinforcement Learning project that demonstrates how an autonomous drone can learn to navigate a 3D environment with obstacles using Q-Learning. The agent is trained to maximize rewards by reaching a target location while avoiding collisions, and the learned policy is visualized in an interactive 3D simulation.

**🔑 Key Features**

- Q-Learning Agent: Implements tabular Q-Learning with epsilon-greedy exploration.
- 3D Grid Environment: Custom environment with static obstacles and defined start/goal states.
- Reward Engineering: Penalties for collisions/out-of-bound moves, positive rewards for reaching the target.
- Training Dashboard:
  - Rewards per episode
  - Smoothed learning curve
  - Steps per episode
  - Epsilon decay visualization
- Path Extraction: Extracts optimal policy and visualizes frequently visited states in 3D.
- Simulation: Uses Ursina Engine to render the drone’s autonomous navigation in a virtual 3D space.

**🧠 Technical Stack**

- Python
- Reinforcement Learning (Q-Learning)
- NumPy, Matplotlib (for training + visualization)
- Ursina Engine (for 3D interactive rendering)

**📊 Learning Results**

Trained over 30,000 episodes with epsilon decay for exploration/exploitation trade-off.
Drone converges to an optimal collision-free path to the goal.
Performance validated via reward trends, path heatmaps, and real-time simulation.

**🚀 Run the Project**

_pip install ursina numpy matplotlib
python Drone.py_
