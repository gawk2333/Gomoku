import os
import csv
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def preprocess_board(board, board_size):
    """Convert 2D board to a 3D one-hot tensor."""
    one_hot = np.zeros((3, board_size, board_size))  # 3 channels for empty, black, white
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 0:  # Empty
                one_hot[0, i, j] = 1
            elif board[i, j] == 1:  # Black
                one_hot[1, i, j] = 1
            elif board[i, j] == 2:  # White
                one_hot[2, i, j] = 1
    return torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
def actions_to_board(actions, board_size, device):
    """
    Converts batch actions of shape [batch_size, 2] into one-hot encoded boards of shape [batch_size, 1, board_size, board_size].
    
    Args:
        actions (torch.Tensor): Tensor of shape [batch_size, 2] where each row is (row, col).
        board_size (int): Size of the board (e.g., 15 for a 15x15 board).
        device (torch.device): Device to perform computation on (e.g., 'cuda' or 'cpu').
        
    Returns:
        torch.Tensor: One-hot encoded boards of shape [batch_size, 1, board_size, board_size].
    """
    batch_size = actions.size(0)
    # Create a tensor of zeros for the board
    boards = torch.zeros(batch_size, 1, board_size, board_size, device=device)
    
    # Flatten the board dimensions to use scatter_
    flat_boards = boards.view(batch_size, -1)  # Shape: [batch_size, board_size * board_size]
    
    # Calculate the flat indices for each action
    flat_indices = actions[:, 0] * board_size + actions[:, 1]  # row * board_size + col
    
    # Use scatter_ to set the correct positions to 1
    flat_boards.scatter_(1, flat_indices.unsqueeze(1), 1)  # Set indices to 1
    
    # Reshape back to [batch_size, 1, board_size, board_size]
    return flat_boards.view(batch_size, 1, board_size, board_size)



class Actor(nn.Module):
    def __init__(self, board_size=15):
        super(Actor, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply CNN layers
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Logits for actions
        return torch.softmax(x.view(-1, self.board_size, self.board_size), dim=-1)  # Apply softmax along the last dimension



class Critic(nn.Module):
    def __init__(self, board_size=15):
        super(Critic, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.action_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * board_size * board_size + 64 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, state, action):
        # Process state through CNN
        x_state = F.relu(self.conv1(state))
        x_state = F.relu(self.conv2(x_state))
        x_state = F.relu(self.conv3(x_state))
        x_state = x_state.view(x_state.size(0), -1)  # Flatten to [batch_size, ...]

        # Process action through CNN
        action = action.unsqueeze(1)  # Add a channel dimension, [batch_size, 1, board_size, board_size]
        x_action = F.relu(self.action_conv(action))
        x_action = x_action.view(x_action.size(0), -1)  # Flatten to [batch_size, ...]

        # Combine state and action features
        combined = torch.cat([x_state, x_action], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(combined))
        return self.fc2(x)  # Output Q-value




class Player:
    def __init__(self, board_size, sign, device, gamma=0.98):
        self.sign = sign
        self.device = device
        self.gamma = gamma
        self.board_size = board_size
        self.step = 0
        self.win = 0
        self.player = 'black' if sign == 1 else 'white'
        self.csv_file = f'training_log({self.player}).csv'

        self.actor = Actor(board_size).to(device)
        self.actor_target = Actor(board_size).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(board_size).to(device)
        self.critic_target = Critic(board_size).to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.0003)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def choose_action(self, board):
        # Convert board to a tensor
        state = torch.tensor(board, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state)  # Get the Q-value predictions
        
        a = logits.squeeze(0).cpu().numpy()  # Convert to numpy array
        mask = (board == 0)  # Mask for legal moves (empty cells)
        
        if not mask.any():
            # No valid moves, return a tensor with all zeros
            action_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            return torch.tensor(action_map, dtype=torch.float, device=self.device)
        
        # Mask invalid moves and set their Q-values to -inf
        masked_a = np.where(mask, a, -np.inf)
        
        # Find the index of the maximum action value
        flat_index = np.argmax(masked_a)
        row, col = np.unravel_index(flat_index, board.shape)
        
        # Create a 15x15 action map with self.sign at the chosen position
        action_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        action_map[row, col] = self.sign  # Set the chosen action to the player's sign
        
        # Convert to a PyTorch tensor and return
        return torch.tensor(action_map, dtype=torch.float, device=self.device),(row,col)


    def learn(self, memories, batch=16):
        if len(memories) < batch:
            return None, None  # Skip update if not enough samples

        sampled_memories = random.sample(memories, batch)
        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp = zip(*sampled_memories)

        # Preprocess states and actions
        b_s = torch.stack(batch_s).float().to(self.device)  # Shape: [batch_size, board_size, board_size]
        b_s = b_s.unsqueeze(1)  # Add channel dimension, now [batch_size, 1, board_size, board_size]

        b_a = torch.stack(batch_a).float().to(self.device)  # Ensure shape [batch_size, board_size, board_size]


        b_r = torch.tensor(batch_r, dtype=torch.float32).view(batch, 1).to(self.device)
        b_ns = torch.stack(batch_ns).float().to(self.device)  # Shape: [batch_size, board_size, board_size]
        b_ns = b_ns.unsqueeze(1)  # Add channel dimension, now [batch_size, 1, board_size, board_size]

        # Critic update
        Q = self.critic(b_s, b_a)
        with torch.no_grad():
            target_action = self.actor_target(b_ns)
            q_target = b_r + self.gamma * self.critic_target(b_ns, target_action)

        q_loss = F.mse_loss(Q, q_target)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Actor update
        actor_actions = self.actor(b_s)
        policy_loss = -self.critic(b_s, actor_actions).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optim.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.step += 1
        return policy_loss.item(), q_loss.item()



    def soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def log(self, policy_loss, qloss, total_reward, reward_components, phase):
        """
        Log episode results or initialize the log file with headers.
        
        :param policy_loss: Policy loss value.
        :param qloss: Q loss value.
        :param reward_components: Dictionary containing total reward and its components (r1, r2, r3, p).
                                Pass `None` to initialize the log file.
        :param phase: Current phase (1, 2, or 3).
        """
        log_file = f"p{phase}_{self.csv_file}"

        if reward_components is not None:  # Logging an episode result
            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                r1 = sum(reward_components['r1'])
                r2 = sum(reward_components['r2'])
                r3 = sum(reward_components['r3'])
                p = sum(reward_components['p'])
                writer.writerow([self.step, policy_loss, qloss, total_reward, r1, r2, r3, p, self.win])
        else:  # Initialize the log file with headers
            with open(log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["steps", "ploss", "qloss", "total_reward", "r1", "r2", "r3", "p", "win"])


    def save(self, phase):
        abspath = os.path.abspath(os.path.dirname(__file__))
        torch.save(self.actor.state_dict(), os.path.join(abspath, f"{self.player}_p{phase}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(abspath, f"{self.player}_p{phase}_critic.pth"))

    def load(self):
        abspath = os.path.abspath(os.path.dirname(__file__))
        actor_path = os.path.join(abspath, f"{self.player}_actor.pth")
        critic_path = os.path.join(abspath, f"{self.player}_critic.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))



class RandomDropper:
    def __init__(self, board_size, sign, device):
        self.board_size = board_size
        self.sign = sign  # Sign of the player (-1 or 1)
        self.device = device  # Device for tensor compatibility

    def choose_action(self, board):
        """
        Select a random valid action and return both the action map
        and the selected (row, col) move.
        """
        # Ensure board is converted to NumPy for valid_moves detection
        board_np = board.cpu().numpy()  # Convert to NumPy array
        
        # Identify valid moves (empty cells)
        valid_moves = np.argwhere(board_np == 0)  # Get indices of all empty cells
        
        if valid_moves.size == 0:  # No valid moves
            action_map = torch.zeros((self.board_size, self.board_size), dtype=torch.float).to(self.device)
            return action_map, (0, 0)  # Default action if no valid moves

        # Randomly select one valid move
        move_index = np.random.randint(valid_moves.shape[0])
        action_row, action_col = valid_moves[move_index]
        # Create an action map with the player's sign
        action_map = torch.zeros((self.board_size, self.board_size), dtype=torch.float).to(self.device)
        action_map[action_row, action_col] = self.sign

        return action_map, (action_row, action_col)




