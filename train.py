import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clash_env import ClashEnv
from coach import LLMCoach

class LLMStrategyCallback(BaseCallback):
    def __init__(self, coach, update_freq=50, verbose=0):
        super(LLMStrategyCallback, self).__init__(verbose)
        self.coach = coach
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        # Panggil LLM Coach setiap update_freq langkah
        if self.n_calls % self.update_freq == 0:
            # Unwrap the environment from DummyVecEnv
            env = self.training_env.envs[0].unwrapped
            
            # Akses state dari custom env
            elixir = env.state[0]
            player_hp = env.state[1]
            enemy_hp = env.state[2]
            current_step = env.current_step
            
            # Dapatkan strategi dari LLM
            strategy = self.coach.get_strategy(player_hp, enemy_hp, elixir, current_step)
            
            # Set mode strategi di environment
            env.set_strategy_mode(strategy)
            
            if self.verbose > 0:
                print(f"[{self.n_calls}] Update Strategi Makro: {strategy}")
                
        return True

def train():
    # Buat direktori untuk model
    os.makedirs("models", exist_ok=True)
    
    # Inisialisasi Environment dan LLM Coach
    env = ClashEnv()
    coach = LLMCoach()
    
    # Buat model PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_clash_tensorboard/")
    
    # Buat Callback untuk hybrid loop (LLM -> Env -> PPO)
    # Misalnya LLM dipanggil setiap 50 steps
    llm_callback = LLMStrategyCallback(coach, update_freq=50, verbose=1)
    
    print("Memulai proses training...")
    # Lakukan training
    model.learn(total_timesteps=1000, callback=llm_callback)
    
    # Simpan checkpoint model
    model.save("models/ppo_clash_bot")
    print("Model disimpan di models/ppo_clash_bot.zip")

if __name__ == "__main__":
    train()
