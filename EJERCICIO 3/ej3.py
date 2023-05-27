import numpy as np

maze = np.array([
    ['S', '0', '0', '0'],
    ['0', 'X', '0', '0'],
    ['0', 'X', '0', '0'],
    ['0', '0', '0', 'E']
])

actions = ['up', 'down', 'left', 'right']

alpha = 0.5
gamma = 0.9
epsilon = 0.1

num_rows, num_cols = maze.shape

Q = np.zeros((num_rows, num_cols, len(actions)))

def get_valid_actions(state):
    row, col = state
    valid_actions = []
    if row > 0 and maze[row-1, col] != 'X':
        valid_actions.append('up')
    if row < num_rows - 1 and maze[row+1, col] != 'X':
        valid_actions.append('down')
    if col > 0 and maze[row, col-1] != 'X':
        valid_actions.append('left')
    if col < num_cols - 1 and maze[row, col+1] != 'X':
        valid_actions.append('right')
    return valid_actions

def select_action(state):
    valid_actions = get_valid_actions(state)
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(valid_actions)
    else:
        row, col = state
        q_values = Q[row, col, :]
        valid_q_values = [q_values[actions.index(action)] for action in valid_actions]
        max_q = np.max(valid_q_values)
        max_indices = np.where(valid_q_values == max_q)[0]
        selected_index = np.random.choice(max_indices)
        return valid_actions[selected_index]

def update_q(state, action, next_state, reward):
    row, col = state
    next_row, next_col = next_state
    q_value = Q[row, col, actions.index(action)]
    max_q = np.max(Q[next_row, next_col, :])
    Q[row, col, actions.index(action)] = q_value + alpha * (reward + gamma * max_q - q_value)

def print_maze(state):
    maze_copy = np.copy(maze)
    maze_copy[state] = 'A'  # Representación visual del agente
    print(maze_copy)

def q_learning():
    num_episodes = 1000

    for episode in range(num_episodes):
        state = (0, 0)
        total_reward = 0

        print("Episodio:", episode+1)
        print("Estado inicial:", state)
        print_maze(state)

        while True:
            action = select_action(state)

            row, col = state
            if action == 'up':
                next_state = (row - 1, col)
            elif action == 'down':
                next_state = (row + 1, col)
            elif action == 'left':
                next_state = (row, col - 1)
            elif action == 'right':
                next_state = (row, col + 1)

            reward = 0
            if maze[next_state] == 'E':
                reward = 10
            elif maze[next_state] == 'X':
                reward = -10

            update_q(state, action, next_state, reward)

            total_reward += reward
            state = next_state

            print("Acción:", action)
            print("Nuevo estado:", state)
            print("Recompensa:", reward)
            print_maze(state)

            if maze[state] == 'E' or maze[state] == 'X':
                break

        print("Recompensa total:", total_reward)
        print()

# Ejecutar el algoritmo de Q-learning
q_learning()
