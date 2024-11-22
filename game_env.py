import memory
import gymnasium as gym
from gymnasium import spaces
import torch
from player import Player, RandomDropper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.board_size = 15
        self.device = device
        self.board = torch.zeros((self.board_size, self.board_size), dtype=torch.float64)
        self.observation_space = spaces.MultiBinary(self.board_size * self.board_size)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.turn_count = 0  
        self.max_turns = self.board_size * self.board_size
        self.GM = memory.Memory(members=[1, -1])

    def reset(self):
        self.board = torch.zeros((self.board_size, self.board_size), dtype=torch.float64)
        self.turn_count = 0  
        return self.board.flatten()

    def step(self, action, player):
        sign = player.sign
        op_sign = -player.sign
        action_indices = torch.nonzero(action == sign, as_tuple=False)
        
        row, col = action_indices[0].tolist()

        self.board[row, col] = player.sign
        self.turn_count += 1  

        # Check if the game is won
        done = self.check_game()

        if done:
            r1, r2, r3, p = 0, 0, 0, 0 
            return self.board, (r1, r2, r3, p, 1.5), done


        r1 = self.count_connections(sign,op_sign, row, col)
        r2 = self.count_blocking_moves(player)
        r3 = self.centrality_reward(row, col)
        p = -self.turn_count / self.max_turns/50 
        # print(f"Debug: turn_count={self.turn_count}, max_turns={self.max_turns}, p={p}")
        basic_reward = r1 + r2 + r3 + p
        return self.board, (r1, r2, r3, p, basic_reward), done




    def count_connections(self, sign, op_sign, row, col):
        reward = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal (/), diagonal (\)

        for dr, dc in directions:
            length = 1  # Include the current stone
            left_blocked, right_blocked = False, False

            # Check in the negative direction (-dr, -dc)
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    left_blocked = True
                    break
                if self.board[r, c] == sign:
                    length += 1
                elif self.board[r, c] == op_sign:
                    left_blocked = True
                    break
                else:
                    break

            # Check in the positive direction (+dr, +dc)
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    right_blocked = True
                    break
                if self.board[r, c] == sign:
                    length += 1
                elif self.board[r, c] == op_sign:
                    right_blocked = True
                    break
                else:
                    break

            # Calculate maximum potential length
            max_length = length + (0 if left_blocked else 1) + (0 if right_blocked else 1)
            if max_length >= 5:
                if length == 4 and not left_blocked and not right_blocked:
                    reward += 0.3  
                elif length == 4 and (left_blocked or right_blocked):
                    reward += 0.2 
                if length == 3 and not left_blocked and not right_blocked:
                    reward += 0.1  
                elif length == 3 and (left_blocked or right_blocked):
                    reward += 0.05 
                elif length == 2:
                    reward += 0.02  
            elif max_length < 5:
                continue

        return reward

    def count_blocking_moves(self, player):
        opponent_sign = -player.sign
        blocking_moves = 0

        def can_block(row, col, dr, dc):
            count = 0
            open_ends = 0

            # Check backward
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == opponent_sign:
                    count += 1
                elif self.board[r, c] == 0:
                    open_ends += 1
                    break
                else:
                    break
                r, c = r - dr, c - dc

            # Check forward
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == opponent_sign:
                    count += 1
                elif self.board[r, c] == 0:
                    open_ends += 1
                    break
                else:
                    break
                r, c = r + dr, c + dc

            return count, open_ends

        def check_double_threes(row, col):
            three_count = 0
            for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count, open_ends = can_block(row, col, dr, dc)
                if count == 2 and open_ends == 2: 
                    three_count += 1
            return three_count >= 2

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0: 
                    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        count, open_ends = can_block(row, col, dr, dc)
                        if count == 3 and open_ends == 2:  
                            blocking_moves += 1
                        elif count == 4 and open_ends >= 1:
                            blocking_moves += 1
                    if check_double_threes(row, col):
                        blocking_moves += 1

        return blocking_moves*0.005


    def centrality_reward(self, row, col):
        """
        Reward for playing closer to the center of the board.
        """
        center = self.board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        max_distance = self.board_size
        return (max_distance - distance_from_center) / max_distance*0.01



    def check_game(self):
        # check for a win
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != 0:
                    if (self.check_line(row, col, 1, 0) or  # Horizontal
                            self.check_line(row, col, 0, 1) or  # Vertical
                            self.check_line(row, col, 1, 1) or  # Diagonal /
                            self.check_line(row, col, 1, -1)):  # Diagonal \
                        return True  # Game win

        # check for a draw (board full and no one has won)
        if not (self.board == 0).any():  
            print("The game is a draw. The board is full.")
            return True  

        return False  


    def check_line(self, row, col, dr, dc):
        sign = self.board[row, col]
        count = 1
        for _ in range(1, 5):
            row += dr
            col += dc
            if 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == sign:
                count += 1
            else:
                break
        return count >= 5


    def train(self, args):
        num_episodes_phase1 = 2000
        num_episodes_phase2 = 2000
        num_episodes_phase3 = 5000

        player1 = Player(self.board_size, 1, self.device)
        player2 = Player(self.board_size, -1, self.device)
    

        random_dropper1 = RandomDropper(self.board_size, -1, self.device)
        random_dropper2 = RandomDropper(self.board_size, 1, self.device)


        def play_episode(_player1, _player2, phase=1):
            self.reset()
            _current_player = _player1
            _opponent = _player2
            total_reward1 = 0
            total_reward2 = 0

            total_reward1_components = {"r1": [], "r2": [], "r3": [], "p": []}
            total_reward2_components = {"r1": [], "r2": [], "r3": [], "p": []}

            if isinstance(_player1, Player):
                _player1.win = 0
                _player1.step = 0
            if isinstance(_player2, Player):
                _player2.win = 0
                _player2.step = 0

            player1_loss = [] if isinstance(_player1, Player) else None
            player2_loss = [] if isinstance(_player2, Player) else None
            done = False

            while not done:
                action_map, _ = _current_player.choose_action(self.board)
                board, (r1, r2, r3, p, reward), done = self.step(action_map, _current_player)

                if done and isinstance(_current_player, Player):
                    _current_player.win = 1

                if _current_player == _player1:
                    total_reward1 += reward
                    total_reward1_components["r1"].append(r1)
                    total_reward1_components["r2"].append(r2)
                    total_reward1_components["r3"].append(r3)
                    total_reward1_components["p"].append(p)
                else:
                    total_reward2 += reward
                    total_reward2_components["r1"].append(r1)
                    total_reward2_components["r2"].append(r2)
                    total_reward2_components["r3"].append(r3)
                    total_reward2_components["p"].append(p)

                self.GM.temp_memory[_current_player.sign]["s"].append(self.board.clone())
                self.GM.temp_memory[_current_player.sign]["a"].append(action_map)
                self.GM.temp_memory[_current_player.sign]["r"].append(reward)

                # Determine fp for the other player
                if len(self.GM.temp_memory[_opponent.sign]["a"]) > 0:
                    fp = self.GM.temp_memory[_opponent.sign]["a"][-1]
                else:
                    fp = torch.zeros((self.board_size, self.board_size), dtype=torch.float).to(self.device)
                self.GM.temp_memory[_current_player.sign]["fp"].append(fp)

                if len(self.GM.temp_memory[_current_player.sign]["s"]) > 2:
                    s = self.GM.temp_memory[_current_player.sign]["s"][-3]
                    ns = self.GM.temp_memory[_current_player.sign]["s"][-2]
                    fp = self.GM.temp_memory[_current_player.sign]["fp"][-2]
                    nfp = self.GM.temp_memory[_current_player.sign]["fp"][-1]
                    a = self.GM.temp_memory[_current_player.sign]["a"][-3]
                    r = self.GM.temp_memory[_current_player.sign]["r"][-2]
                    self.GM.remember(s, fp, a, r, ns, nfp, _current_player.sign)

                if isinstance(_current_player, Player):
                    p_loss, q_loss = _current_player.learn(self.GM.memory[_current_player.sign])
                    if p_loss is not None and q_loss is not None:
                        if _current_player == _player1:
                            player1_loss.append([p_loss, q_loss])
                        else:
                            player2_loss.append([p_loss, q_loss])

                _current_player, _opponent = _opponent, _current_player

                player1_mean_ploss = (
                    sum(loss[0] for loss in player1_loss) / len(player1_loss) if player1_loss else 0.0
                )
                player1_mean_qloss = (
                    sum(loss[1] for loss in player1_loss) / len(player1_loss) if player1_loss else 0.0
                )
                player2_mean_ploss = (
                    sum(loss[0] for loss in player2_loss) / len(player2_loss) if player2_loss else 0.0
                )
                player2_mean_qloss = (
                    sum(loss[1] for loss in player2_loss) / len(player2_loss) if player2_loss else 0.0
                )

            if isinstance(_player1, Player):
                _player1.log(
                    player1_mean_ploss, 
                    player1_mean_qloss, 
                    total_reward1,
                    total_reward1_components,
                    phase
                )
                _player1.save(phase)

            if isinstance(_player2, Player):
                _player2.log(
                    player2_mean_ploss, 
                    player2_mean_qloss, 
                    total_reward2,
                    total_reward2_components,
                    phase
                )
                _player2.save(phase)
            return (
                total_reward1,
                total_reward2,
                player1_mean_ploss,
                player1_mean_qloss,
                player2_mean_ploss,
                player2_mean_qloss,
                total_reward1_components,
                total_reward2_components,
            )


        player1.log(None, None, None, None, phase=1)
        for episode in range(num_episodes_phase1):
            reward1, reward2, p1_ploss, p1_qloss, p2_ploss, p2_qloss, reward1_components, reward2_components = play_episode(player1, random_dropper1, phase=1)
            print(f"Phase 1 - Episode {episode + 1}/{num_episodes_phase1}")
            print(f"Player 1 Total Reward: {reward1:.2f}, Win: {player1.win}")
            print(f"  Reward Components: r1: {sum(reward1_components['r1']):.2f}, r2: {sum(reward1_components['r2']):.2f}, r3: {sum(reward1_components['r3']):.2f}, p: {sum(reward1_components['p']):.2f}")
            print(f"  P Loss: {p1_ploss:.4f}, Q Loss: {p1_qloss:.4f}")

        player2.log(None, None, None, None, phase=2)
        for episode in range(num_episodes_phase2):
            reward1, reward2, p1_ploss, p1_qloss, p2_ploss, p2_qloss, reward1_components, reward2_components = play_episode(random_dropper2, player2, phase=2)
            print(f"Phase 2 - Episode {episode + 1}/{num_episodes_phase2}")
            print(f"Player 2 Total Reward: {reward2:.2f}, Win: {player2.win}")
            print(f"  Reward Components: r1: {sum(reward2_components['r1']):.2f}, r2: {sum(reward2_components['r2']):.2f}, r3: {sum(reward2_components['r3']):.2f}, p: {sum(reward2_components['p']):.2f}")
            print(f"  P Loss: {p2_ploss:.4f}, Q Loss: {p2_qloss:.4f}")

        player1.log(None, None, None, None, phase=3)
        player2.log(None, None, None, None, phase=3)
        for episode in range(num_episodes_phase3):
            reward1, reward2, p1_ploss, p1_qloss, p2_ploss, p2_qloss, reward1_components, reward2_components = play_episode(player1, player2, phase=3)
            print(f"Phase 3 - Episode {episode + 1}/{num_episodes_phase3}")
            print(f"Player 1 Total Reward: {reward1:.2f}, Win: {player1.win}")
            print(f"  Reward Components: r1: {sum(reward1_components['r1']):.2f}, r2: {sum(reward1_components['r2']):.2f}, r3: {sum(reward1_components['r3']):.2f}, p: {sum(reward1_components['p']):.2f}")
            print(f"  P Loss: {p1_ploss:.4f}, Q Loss: {p1_qloss:.4f}")
            print(f"Player 2 Total Reward: {reward2:.2f}, Win: {player2.win}")
            print(f"  Reward Components: r1: {sum(reward2_components['r1']):.2f}, r2: {sum(reward2_components['r2']):.2f}, r3: {sum(reward2_components['r3']):.2f}, p: {sum(reward2_components['p']):.2f}")
            print(f"  P Loss: {p2_ploss:.4f}, Q Loss: {p2_qloss:.4f}")
