import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ClashEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    This simulates a simplified Clash Royale game.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(ClashEnv, self).__init__()
        
        # Action space: 0 (Wait), 1 (Cheap Unit - Cost 2), 2 (Heavy Push - Cost 5)
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # [0]: Elixir pemain (0 - 10)
        # [1]: HP Tower pemain (0 - 3000)
        # [2]: HP Tower musuh (0 - 3000)
        # [3]: Current Strategy Mode (0: Balanced, 1: Aggressive, 2: Defensive)
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([10, 3000, 3000, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Maximum steps per episode
        self.max_steps = 200
        self.current_step = 0
        
        # Strategy mode string mapping for convenience
        self.strategy_map = {0: "BALANCED", 1: "AGGRESSIVE", 2: "DEFENSIVE"}
        
        # Initialize state
        self.reset()

    def set_strategy_mode(self, mode_str):
        """Update current strategy mode from LLM Coach."""
        mode_str = mode_str.upper().strip()
        if mode_str == "AGGRESSIVE":
            self.state[3] = 1.0
        elif mode_str == "DEFENSIVE":
            self.state[3] = 2.0
        else: # Default to BALANCED
            self.state[3] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initial state: 5 elixir, 3000 HP towers, Balanced mode
        self.state = np.array([5.0, 3000.0, 3000.0, 0.0], dtype=np.float32)
        
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        elixir = self.state[0]
        player_hp = self.state[1]
        enemy_hp = self.state[2]
        strategy_mode = int(self.state[3])
        
        damage_dealt = 0.0
        damage_taken = 0.0
        reward = 0.0
        
        # Environment step simulation (opponent action logic - simplified)
        # Opponent deals random damage each step
        enemy_attack = np.random.uniform(10, 50)
        
        # Action logic
        if action == 1: # Cheap Unit
            if elixir >= 2.0:
                self.state[0] -= 2.0
                damage_dealt = np.random.uniform(50, 150)
            else:
                reward -= 10 # Penalty for invalid action
        elif action == 2: # Heavy Push
            if elixir >= 5.0:
                self.state[0] -= 5.0
                damage_dealt = np.random.uniform(150, 400)
            else:
                reward -= 10 # Penalty for invalid action
                
        # Natural elixir regeneration
        self.state[0] = min(10.0, self.state[0] + 0.5)
        
        # Enemy counter-attack based on our push
        if damage_dealt > 0:
            damage_taken = enemy_attack + np.random.uniform(0, 50) # Retaliation
        else:
            damage_taken = enemy_attack
            
        # Update HP
        self.state[1] = max(0.0, player_hp - damage_taken)
        self.state[2] = max(0.0, enemy_hp - damage_dealt)
        
        # Modulasi reward berdasarkan Strategy Mode
        if strategy_mode == 0: # Balanced
            reward += (damage_dealt - damage_taken)
        elif strategy_mode == 1: # Aggressive
            reward += (2.0 * damage_dealt - 0.5 * damage_taken)
        elif strategy_mode == 2: # Defensive
            reward += (0.5 * damage_dealt - 2.0 * damage_taken)
            
        # Check termination
        terminated = bool(self.state[1] <= 0 or self.state[2] <= 0)
        truncated = bool(self.current_step >= self.max_steps)
        
        if terminated:
            if self.state[2] <= 0 and self.state[1] > 0:
                reward += 1000 # Win bonus
            elif self.state[1] <= 0:
                reward -= 1000 # Lose penalty
                
        info = {
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "strategy": self.strategy_map[strategy_mode]
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step} | Elixir: {self.state[0]:.1f} | "
              f"Player HP: {self.state[1]:.0f} | Enemy HP: {self.state[2]:.0f} | "
              f"Mode: {self.strategy_map[int(self.state[3])]}")
