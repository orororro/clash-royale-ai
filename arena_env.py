import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Unit:
    def __init__(self, unit_type, x, y, team):
        self.unit_type = unit_type  # 1: Knight, 2: Archer
        self.x = float(x)
        self.y = float(y)
        self.team = team  # "agent" or "enemy"
        
        if unit_type == 1:  # Knight (Melee Tank/Brawler)
            self.max_hp = 600.0
            self.hp = 600.0
            self.damage = 60.0
            self.attack_range = 1.3
            self.speed = 0.5
            self.attack_interval = 2  # steps between attacks
            self.cost = 3
            self.name = "Knight"
        elif unit_type == 2:  # Archer (Ranged DPS)
            self.max_hp = 250.0
            self.hp = 250.0
            self.damage = 35.0
            self.attack_range = 5.0
            self.speed = 0.5
            self.attack_interval = 2
            self.cost = 3
            self.name = "Archer"
            
        self.cooldown = 0
        self.is_alive = True

    def distance_to(self, target_x, target_y):
        return math.hypot(self.x - target_x, self.y - target_y)


class Tower:
    def __init__(self, x, y, team, max_hp=1500.0):
        self.x = float(x)
        self.y = float(y)
        self.team = team  # "agent" or "enemy"
        self.max_hp = float(max_hp)
        self.hp = float(max_hp)
        self.attack_range = 6.0
        self.damage = 40.0
        self.attack_interval = 2
        self.cooldown = 0
        self.is_alive = True

    def distance_to(self, target_x, target_y):
        return math.hypot(self.x - target_x, self.y - target_y)


class ArenaEnv(gym.Env):
    """
    2D Grid Simulator Arena for Clash Royale AI.
    - Grid: 18 (Width) x 32 (Height)
    - River at y=15..16, Bridges at Left (x=3..4) and Right (x=13..14)
    - Agent Half: y in [16..31], Enemy Half: y in [0..15]
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super(ArenaEnv, self).__init__()
        
        self.render_mode = render_mode
        self.grid_width = 18
        self.grid_height = 32
        self.cell_size = 24  # pixels per cell for pygame
        
        # Action space: MultiDiscrete([card_type (0: Wait, 1: Knight, 2: Archer), pos_x (0..17), pos_y (0..31)])
        self.action_space = spaces.MultiDiscrete([3, self.grid_width, self.grid_height])
        
        # Observation space:
        # [0]: Agent Elixir (0..10)
        # [1]: Enemy Elixir (0..10)
        # [2]: Agent Tower Left HP (0..1500)
        # [3]: Agent Tower Right HP (0..1500)
        # [4]: Enemy Tower Left HP (0..1500)
        # [5]: Enemy Tower Right HP (0..1500)
        # [6]: Strategy Mode (0: Balanced, 1: Aggressive, 2: Defensive)
        # [7]: Agent Unit Count (0..50)
        # [8]: Enemy Unit Count (0..50)
        # [9]: Agent Knight Count
        # [10]: Agent Archer Count
        # [11]: Enemy Knight Count
        # [12]: Enemy Archer Count
        # [13]: Step progress (0..1)
        low_obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        high_obs = np.array([10, 10, 1500, 1500, 1500, 1500, 2, 50, 50, 50, 50, 50, 50, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.bridges = [(3.5, 15.5), (13.5, 15.5)]  # Left & Right bridge waypoints
        self.max_steps = 300
        self.current_step = 0
        
        self.strategy_mode = 0  # 0: Balanced, 1: Aggressive, 2: Defensive
        self.strategy_map = {0: "BALANCED", 1: "AGGRESSIVE", 2: "DEFENSIVE"}
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.font = None
        
        self.units = []
        self.towers = []
        self.agent_elixir = 5.0
        self.enemy_elixir = 5.0
        
        self.reset()

    def set_strategy_mode(self, mode_str_or_int):
        if isinstance(mode_str_or_int, str):
            mode = mode_str_or_int.upper().strip()
            if mode == "AGGRESSIVE":
                self.strategy_mode = 1
            elif mode == "DEFENSIVE":
                self.strategy_mode = 2
            else:
                self.strategy_mode = 0
        else:
            self.strategy_mode = int(mode_str_or_int)

    def _get_obs(self):
        agent_knights = sum(1 for u in self.units if u.team == "agent" and u.unit_type == 1)
        agent_archers = sum(1 for u in self.units if u.team == "agent" and u.unit_type == 2)
        enemy_knights = sum(1 for u in self.units if u.team == "enemy" and u.unit_type == 1)
        enemy_archers = sum(1 for u in self.units if u.team == "enemy" and u.unit_type == 2)
        
        agent_units = agent_knights + agent_archers
        enemy_units = enemy_knights + enemy_archers
        
        step_prog = min(1.0, float(self.current_step) / float(self.max_steps))
        
        obs = np.array([
            self.agent_elixir,
            self.enemy_elixir,
            self.agent_tower_l.hp,
            self.agent_tower_r.hp,
            self.enemy_tower_l.hp,
            self.enemy_tower_r.hp,
            float(self.strategy_mode),
            float(agent_units),
            float(enemy_units),
            float(agent_knights),
            float(agent_archers),
            float(enemy_knights),
            float(enemy_archers),
            step_prog
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.agent_elixir = 5.0
        self.enemy_elixir = 5.0
        
        # Initialize Towers
        # Enemy towers at top (y=5.5)
        self.enemy_tower_l = Tower(3.5, 5.5, team="enemy", max_hp=1500.0)
        self.enemy_tower_r = Tower(13.5, 5.5, team="enemy", max_hp=1500.0)
        # Agent towers at bottom (y=26.5)
        self.agent_tower_l = Tower(3.5, 26.5, team="agent", max_hp=1500.0)
        self.agent_tower_r = Tower(13.5, 26.5, team="agent", max_hp=1500.0)
        
        self.towers = [self.enemy_tower_l, self.enemy_tower_r, self.agent_tower_l, self.agent_tower_r]
        self.units = []
        
        return self._get_obs(), {}

    def _spawn_unit(self, unit_type, x, y, team):
        unit = Unit(unit_type, x, y, team)
        self.units.append(unit)

    def _find_target(self, unit):
        """Find closest enemy unit or enemy tower."""
        opp_team = "enemy" if unit.team == "agent" else "agent"
        candidates = []
        
        # Check enemy units
        for u in self.units:
            if u.team == opp_team and u.is_alive:
                candidates.append((u.distance_to(unit.x, unit.y), u))
                
        # Check enemy towers
        for t in self.towers:
            if t.team == opp_team and t.is_alive:
                candidates.append((t.distance_to(unit.x, unit.y), t))
                
        if not candidates:
            return None
            
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _move_unit_towards(self, unit, target):
        """Move unit towards target or bridge if crossing river is required."""
        target_x, target_y = target.x, target.y
        
        # Check river crossing requirement (river is between y=14.5 and y=16.5)
        needs_crossing = (unit.team == "agent" and unit.y > 16.0 and target_y < 15.0) or \
                         (unit.team == "enemy" and unit.y < 15.0 and target_y > 16.0)
                         
        if needs_crossing:
            # Choose nearest bridge
            b1 = self.bridges[0]
            b2 = self.bridges[1]
            bridge = b1 if unit.distance_to(*b1) < unit.distance_to(*b2) else b2
            
            # If not yet on bridge, head towards bridge
            if abs(unit.y - bridge[1]) > 0.8:
                target_x, target_y = bridge[0], bridge[1]
                
        dx = target_x - unit.x
        dy = target_y - unit.y
        dist = math.hypot(dx, dy)
        
        if dist > 0.01:
            move_step = min(unit.speed, dist)
            unit.x += (dx / dist) * move_step
            unit.y += (dy / dist) * move_step
            
            # Clamp inside grid boundaries
            unit.x = max(0.5, min(self.grid_width - 0.5, unit.x))
            unit.y = max(0.5, min(self.grid_height - 0.5, unit.y))

    def step(self, action):
        self.current_step += 1
        card_type, pos_x, pos_y = int(action[0]), float(action[1]), float(action[2])
        
        reward = 0.0
        damage_dealt = 0.0
        damage_taken = 0.0
        
        # 1. Elixir Regeneration
        self.agent_elixir = min(10.0, self.agent_elixir + 0.15)
        self.enemy_elixir = min(10.0, self.enemy_elixir + 0.15)
        
        # 2. Process Agent Action
        if card_type in [1, 2]:
            cost = 3  # Both Knight and Archer cost 3 elixir
            # Agent can only place units on their side (y >= 16)
            if self.agent_elixir >= cost and pos_y >= 16.0:
                self.agent_elixir -= cost
                self._spawn_unit(card_type, pos_x + 0.5, pos_y + 0.5, team="agent")
                reward += 5.0  # Reward positif saat berhasil drop unit
            else:
                # Jika Elixir tidak cukup atau illegal, ubah aksi jadi Wait otomatis
                card_type = 0
                
        # Evaluasi Wait action (Leaking Penalty)
        if card_type == 0:
            if self.agent_elixir >= 10.0:
                reward -= 0.1  # Penalti bocor elixir agar tidak terus-terusan diam
                
        # 3. Simple Enemy AI logic
        if self.enemy_elixir >= 3.0 and np.random.rand() < 0.15:
            e_card = np.random.choice([1, 2])
            e_x = np.random.choice([3.5, 13.5]) + np.random.uniform(-1.0, 1.0)
            e_y = np.random.uniform(7.0, 12.0)
            self.enemy_elixir -= 3.0
            self._spawn_unit(e_card, e_x, e_y, team="enemy")
            
        # 4. Units Actions (Movement & Combat)
        for unit in self.units:
            if not unit.is_alive:
                continue
                
            unit.cooldown = max(0, unit.cooldown - 1)
            target = self._find_target(unit)
            
            if target is None:
                continue
                
            dist = unit.distance_to(target.x, target.y)
            
            if dist <= unit.attack_range:
                # In attack range -> Attack
                if unit.cooldown == 0:
                    target.hp -= unit.damage
                    unit.cooldown = unit.attack_interval
                    
                    if unit.team == "agent":
                        damage_dealt += unit.damage
                        if isinstance(target, Tower):
                            reward += unit.damage * 0.1  # Reward ekstra proporsional jika nyerang tower musuh
                    else:
                        damage_taken += unit.damage
                        
                    if target.hp <= 0:
                        target.hp = 0
                        target.is_alive = False
            else:
                # Out of range -> Move
                self._move_unit_towards(unit, target)
                
        # 5. Towers Combat
        for tower in self.towers:
            if not tower.is_alive:
                continue
                
            tower.cooldown = max(0, tower.cooldown - 1)
            opp_team = "enemy" if tower.team == "agent" else "agent"
            
            # Find closest enemy unit in tower range
            in_range_enemies = [
                u for u in self.units 
                if u.team == opp_team and u.is_alive and tower.distance_to(u.x, u.y) <= tower.attack_range
            ]
            
            if in_range_enemies and tower.cooldown == 0:
                in_range_enemies.sort(key=lambda u: tower.distance_to(u.x, u.y))
                target_unit = in_range_enemies[0]
                target_unit.hp -= tower.damage
                tower.cooldown = tower.attack_interval
                
                if tower.team == "agent":
                    damage_dealt += tower.damage
                else:
                    damage_taken += tower.damage
                    
                if target_unit.hp <= 0:
                    target_unit.hp = 0
                    target_unit.is_alive = False

        # 6. Clean up dead units
        self.units = [u for u in self.units if u.is_alive]
        
        # 7. Modulate Reward by Strategy Mode
        if self.strategy_mode == 0:  # Balanced
            reward += (damage_dealt * 0.1 - damage_taken * 0.1)
        elif self.strategy_mode == 1:  # Aggressive
            reward += (damage_dealt * 0.2 - damage_taken * 0.05)
        elif self.strategy_mode == 2:  # Defensive
            reward += (damage_dealt * 0.05 - damage_taken * 0.2)
            
        # 8. Check Match Termination
        agent_alive_towers = sum(1 for t in [self.agent_tower_l, self.agent_tower_r] if t.is_alive)
        enemy_alive_towers = sum(1 for t in [self.enemy_tower_l, self.enemy_tower_r] if t.is_alive)
        
        terminated = bool(agent_alive_towers == 0 or enemy_alive_towers == 0)
        truncated = bool(self.current_step >= self.max_steps)
        
        if terminated:
            if enemy_alive_towers == 0 and agent_alive_towers > 0:
                reward += 500.0  # Victory
            elif agent_alive_towers == 0:
                reward -= 500.0  # Defeat
                
        info = {
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "strategy": self.strategy_map[self.strategy_mode]
        }
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
            
        try:
            import pygame
        except ImportError:
            return
            
        width = self.grid_width * self.cell_size
        height = self.grid_height * self.cell_size
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Clash Royale AI Arena 2D")
            else:
                self.screen = pygame.Surface((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 12)
            self.title_font = pygame.font.SysFont("Arial", 14, bold=True)
            
        # Background: Arena Grass
        self.screen.fill((118, 186, 75))
        
        # River (y = 15..16)
        river_rect = pygame.Rect(0, 15 * self.cell_size, width, 2 * self.cell_size)
        pygame.draw.rect(self.screen, (70, 150, 230), river_rect)
        
        # Bridges: Left (x=3..5) & Right (x=13..15)
        b_width = 2 * self.cell_size
        b_height = 2 * self.cell_size
        b1_rect = pygame.Rect(3 * self.cell_size, 15 * self.cell_size, b_width, b_height)
        b2_rect = pygame.Rect(13 * self.cell_size, 15 * self.cell_size, b_width, b_height)
        pygame.draw.rect(self.screen, (160, 110, 60), b1_rect)
        pygame.draw.rect(self.screen, (160, 110, 60), b2_rect)
        
        # Draw Arena Lines (Subtle Grid)
        for gx in range(self.grid_width + 1):
            pygame.draw.line(self.screen, (100, 160, 65), (gx * self.cell_size, 0), (gx * self.cell_size, height), 1)
        for gy in range(self.grid_height + 1):
            pygame.draw.line(self.screen, (100, 160, 65), (0, gy * self.cell_size), (width, gy * self.cell_size), 1)
            
        # Draw Towers
        for tower in self.towers:
            if not tower.is_alive:
                continue
            center_x = int(tower.x * self.cell_size)
            center_y = int(tower.y * self.cell_size)
            color = (50, 100, 230) if tower.team == "agent" else (220, 50, 50)
            
            # Tower base
            tower_radius = int(self.cell_size * 1.2)
            pygame.draw.circle(self.screen, color, (center_x, center_y), tower_radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), tower_radius, 2)
            
            # Tower HP bar
            hp_pct = max(0.0, tower.hp / tower.max_hp)
            bar_w = 40
            bar_h = 6
            bar_x = center_x - bar_w // 2
            bar_y = center_y - tower_radius - 10
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 230, 80), (bar_x, bar_y, int(bar_w * hp_pct), bar_h))
            
        # Draw Units
        for unit in self.units:
            if not unit.is_alive:
                continue
            u_x = int(unit.x * self.cell_size)
            u_y = int(unit.y * self.cell_size)
            u_color = (0, 180, 255) if unit.team == "agent" else (255, 70, 70)
            u_radius = 10 if unit.unit_type == 1 else 7  # Knight larger than Archer
            
            pygame.draw.circle(self.screen, u_color, (u_x, u_y), u_radius)
            
            # Label Unit (K = Knight, A = Archer)
            txt = "K" if unit.unit_type == 1 else "A"
            lbl = self.font.render(txt, True, (255, 255, 255))
            self.screen.blit(lbl, (u_x - 4, u_y - 6))
            
            # HP bar for unit
            u_hp_pct = max(0.0, unit.hp / unit.max_hp)
            u_bar_w = 20
            u_bar_h = 3
            pygame.draw.rect(self.screen, (40, 40, 40), (u_x - 10, u_y - u_radius - 6, u_bar_w, u_bar_h))
            pygame.draw.rect(self.screen, (0, 255, 50), (u_x - 10, u_y - u_radius - 6, int(u_bar_w * u_hp_pct), u_bar_h))
            
        # Top HUD Info Overlay
        hud_bg = pygame.Surface((width, 52), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 160))
        self.screen.blit(hud_bg, (0, 0))
        
        info_text = f"Step: {self.current_step}/{self.max_steps} | Mode: {self.strategy_map[self.strategy_mode]}"
        info_surf = self.title_font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surf, (8, 4))
        
        elixir_text = f"Agent Elixir: {self.agent_elixir:.1f} | Enemy: {self.enemy_elixir:.1f}"
        elixir_surf = self.font.render(elixir_text, True, (255, 220, 50))
        self.screen.blit(elixir_surf, (8, 20))
        
        # Tambahan info untuk Live Training (Total Timesteps & Reward)
        c_ts = getattr(self, 'custom_timesteps', 0)
        c_rew = getattr(self, 'custom_reward', 0.0)
        if c_ts > 0 or c_rew != 0.0:
            tr_text = f"Total TS: {c_ts} | Cur Reward: {c_rew:.1f}"
            tr_surf = self.font.render(tr_text, True, (0, 255, 255))
            self.screen.blit(tr_surf, (8, 36))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
