from ursina import *
import numpy as np
import random
from time import time as pytime
import matplotlib.pyplot as plt

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Parameters
GRID_SIZE = 10
OBSTACLE_POSITIONS = [
    (2, 2, 2), (2, 3, 2), (2, 4, 2),
    (5, 5, 5), (5, 6, 5), (6, 5, 5),
    (8, 1, 3), (8, 1, 4), (8, 1, 5),
    (3, 8, 8), (4, 8, 8),
]
START_STATE = (0, 0, 0)
TARGET_STATE = (GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1)

# Q-Learning parameters
Q_TABLE = {}
ACTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.01
TOTAL_EPISODES = 30000

# Tracking variables
episode_rewards = []
episode_steps = []
epsilon_values = []

def get_q_values(state):
    if state not in Q_TABLE:
        Q_TABLE[state] = np.zeros(len(ACTIONS))
    return Q_TABLE[state]

def take_action(state, action_index):
    action = ACTIONS[action_index]
    new_state = tuple(np.add(state, action))

    if not all(0 <= coord < GRID_SIZE for coord in new_state):
        return state, -100, True

    if new_state in OBSTACLE_POSITIONS:
        return new_state, -100, True

    if new_state == TARGET_STATE:
        return new_state, 100, True

    return new_state, -1, False

def train():
    global EPSILON
    print("Training started...")
    for ep in range(TOTAL_EPISODES):
        state = START_STATE
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 500:
            if random.random() < EPSILON:
                action = random.randint(0, len(ACTIONS) - 1)
            else:
                action = np.argmax(get_q_values(state))

            new_state, reward, done = take_action(state, action)
            total_reward += reward

            old_q = get_q_values(state)[action]
            future_q = np.max(get_q_values(new_state))
            new_q = old_q + LEARNING_RATE * (reward + DISCOUNT * future_q - old_q)
            get_q_values(state)[action] = new_q
            state = new_state
            steps += 1

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilon_values.append(EPSILON)

    print("Training complete.")

# Run training
train()

# --- Plot Learning Dashboard ---
window = 200
smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(episode_rewards)
axs[0, 0].set_title("Rewards per Episode")
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Reward")

axs[0, 1].plot(smoothed_rewards, color='orange')
axs[0, 1].set_title(f"Smoothed Reward (window={window})")
axs[0, 1].set_xlabel("Episode")
axs[0, 1].set_ylabel("Smoothed Reward")

axs[1, 0].plot(episode_steps, color='green')
axs[1, 0].set_title("Steps per Episode")
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Steps")

axs[1, 1].plot(epsilon_values, color='red')
axs[1, 1].set_title("Epsilon Decay")
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Epsilon")

plt.tight_layout()
plt.show()

# --- Extract Optimal Path ---
path = []
state = START_STATE
done = False
visited = set()
max_path_steps = 300
steps = 0
path_visit_count = {}

while not done and state not in visited and steps < max_path_steps:
    visited.add(state)
    path.append(state)
    path_visit_count[state] = path_visit_count.get(state, 0) + 1
    action = np.argmax(get_q_values(state))
    state, _, done = take_action(state, action)
    steps += 1

if state == TARGET_STATE:
    path.append(TARGET_STATE)
    path_visit_count[TARGET_STATE] = path_visit_count.get(TARGET_STATE, 0) + 1

# --- 3D Path Heatmap ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs, colors = [], [], [], []

max_visits = max(path_visit_count.values())
for (x, y, z), count in path_visit_count.items():
    xs.append(x)
    ys.append(y)
    zs.append(z)
    colors.append(count / max_visits)

img = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', s=100)
ax.set_title("3D Path Heatmap (Visit Frequency)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.colorbar(img, label='Relative Visit Frequency')
plt.show()

# --- Ursina Visualization ---
app = Ursina()

CELL_SIZE = 1.0
GRID_CENTER = GRID_SIZE // 2

def world_pos(pos):
    return Vec3(
        (pos[0] - GRID_CENTER) * CELL_SIZE,
        (pos[1] - GRID_CENTER) * CELL_SIZE,
        (pos[2] - GRID_CENTER) * CELL_SIZE
    )

Entity(
    model='plane',
    scale=(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE),
    texture='white_cube',
    texture_scale=(GRID_SIZE, GRID_SIZE),
    color=color.light_gray,
    y=-(GRID_CENTER + 0.5) * CELL_SIZE
)

for pos in OBSTACLE_POSITIONS:
    Entity(model='cube', position=world_pos(pos), color=color.red, scale=CELL_SIZE * 0.5)

Entity(model='cube', position=world_pos(TARGET_STATE), color=color.green, scale=CELL_SIZE * 0.5)

drone = Entity(model='sphere', position=world_pos(START_STATE), color=color.cyan, scale=CELL_SIZE * 0.5)

EditorCamera()
center_world = world_pos((5, 10, 10))
cam_dist = GRID_SIZE * CELL_SIZE * 2
camera.position = center_world + Vec3(cam_dist, cam_dist, -cam_dist)
camera.look_at(center_world)
camera.rotation = (45, -45, 0)

for pos in path:
    Entity(model='sphere', position=world_pos(pos), scale=CELL_SIZE * 0.18, color=color.azure)

path_index = 0
path_displayed = False
success_text = None
animation_start_time = None

def update():
    global path_index, path_displayed, success_text, animation_start_time
    if animation_start_time is None:
        animation_start_time = pytime()
        return
    if pytime() - animation_start_time < 3.0:
        return
    if path_index < len(path):
        next_pos = world_pos(path[path_index])
        drone.position = lerp(drone.position, next_pos, time.dt * 4)
        if distance(drone.position, next_pos) < 0.1:
            drone.position = next_pos
            path_index += 1
    elif not path_displayed:
        path_displayed = True
        if drone.position == world_pos(TARGET_STATE):
            success_text = Text("Target reached!", origin=(0, -4), scale=2, color=color.green)
        else:
            success_text = Text("Path incomplete.", origin=(0, -4), scale=2, color=color.orange)

app.run()
