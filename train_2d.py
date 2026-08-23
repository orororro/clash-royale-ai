import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from arena_env import ArenaEnv
from coach import LLMCoach

class MacroStrategyCallback(BaseCallback):
    """
    Callback untuk memperbarui mode strategi makro secara periodik selama training 2D.
    """
    def __init__(self, coach, update_freq=200, verbose=0):
        super(MacroStrategyCallback, self).__init__(verbose)
        self.coach = coach
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            env = self.training_env.envs[0].unwrapped
            
            # Ambil status tower & elixir
            player_hp = (env.agent_tower_l.hp + env.agent_tower_r.hp) / 2.0
            enemy_hp = (env.enemy_tower_l.hp + env.enemy_tower_r.hp) / 2.0
            elixir = env.agent_elixir
            current_step = env.current_step
            
            # Tentukan strategi dari Coach
            strategy = self.coach.get_macro_strategy(player_hp, enemy_hp, elixir, current_step)
            env.set_strategy_mode(strategy)
            
            if self.verbose > 0:
                print(f"[Coach] Step {self.n_calls}: Mode diperbarui -> {strategy}")
                
        return True

def train():
    print("=== Memulai Training PPO untuk Arena 2D ===")
    
    # 1. Inisialisasi Environment tanpa render (Headless mode)
    env = ArenaEnv(render_mode=None)
    coach = LLMCoach(model_name="gemini-3.1-flash-lite")
    
    # 2. Setup Model PPO
    # Menggunakan MultiDiscrete action space dengan arsitektur MlpPolicy
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # 3. Setup Callbacks
    macro_cb = MacroStrategyCallback(coach=coach, update_freq=200, verbose=0)
    
    total_timesteps = 50000
    print(f"Target total timesteps: {total_timesteps}")
    start_time = time.time()
    
    # 4. Training Loop
    model.learn(total_timesteps=total_timesteps, callback=macro_cb)
    
    elapsed = time.time() - start_time
    print(f"Training selesai dalam {elapsed:.2f} detik.")
    
    # 5. Simpan Model Terlatih
    model_path = "clash_2d_ppo"
    model.save(model_path)
    print(f"Model berhasil disimpan ke {model_path}.zip")

if __name__ == "__main__":
    train()
