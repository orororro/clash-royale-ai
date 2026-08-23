import os
import sys
import signal
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from arena_env import ArenaEnv
from coach import LLMCoach

class LiveViewCallback(BaseCallback):
    def __init__(self, coach, verbose=0):
        super().__init__(verbose)
        self.coach = coach
        self.ep_reward = 0.0
        self.last_strategy = "BALANCED"
        
    def _on_step(self) -> bool:
        # Dapatkan instance environment asli
        env = self.training_env.envs[0].unwrapped
        
        # Akumulasi reward
        self.ep_reward += self.locals["rewards"][0]
        
        # Sinkronisasi state & minta mode taktik baru setiap 100 steps
        if env.current_step % 100 == 0:
            summary = {
                "my_tower_hp": env.agent_tower_l.hp + env.agent_tower_r.hp,
                "enemy_tower_hp": env.enemy_tower_l.hp + env.enemy_tower_r.hp,
                "avg_elixir": env.agent_elixir
            }
            self.last_strategy = self.coach.get_macro_strategy(
                summary["my_tower_hp"], 
                summary["enemy_tower_hp"], 
                summary["avg_elixir"], 
                self.num_timesteps
            )
            env.set_strategy_mode(self.last_strategy)
            
        # Reset reward tracking jika episode selesai
        if self.locals["dones"][0]:
            self.ep_reward = 0.0
            
        # Handle event Pygame (untuk safe shutdown saat tombol X diklik)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n[!] Jendela Pygame ditutup. Menghentikan training...")
                return False
                
        # Injeksi data ke env agar bisa di-render oleh Pygame di arena_env.py
        env.custom_timesteps = self.num_timesteps
        env.custom_reward = self.ep_reward
        
        # Jeda clock Pygame agar visual mulus (30 FPS)
        env.clock.tick(30)
        
        return True

def main():
    # 1. Inisialisasi LLM Coach
    coach = LLMCoach(
        api_key="sk-1d47fd2534d0197b-8whljv-cfeeba41",
        base_url="http://localhost:20128/v1",
        model_name="gemini-3.1-flash-lite"
    )
    
    # 2. Inisialisasi Environment dengan render_mode='human'
    env = ArenaEnv(render_mode="human")
    
    # 3. Load Model
    MODEL_PATH = "clash_2d_ppo.zip"
    if os.path.exists(MODEL_PATH):
        print(f"[*] Melanjutkan training dari {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("[*] Membuat model PPO baru...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        
    # 4. Handler untuk Safe Shutdown via Ctrl+C
    def signal_handler(sig, frame):
        print("\n[!] Perintah STOP diterima (Ctrl+C). Menyimpan model...")
        model.save(MODEL_PATH)
        print("✅ Model berhasil disimpan. Keluar...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("🚀 LIVE TRAINING DIMULAI")
    print("👉 Tutup jendela Pygame atau tekan Ctrl+C untuk STOP & SAVE")
    print("="*60 + "\n")
    
    # Mulai training dengan callback
    live_callback = LiveViewCallback(coach=coach)
    
    try:
        model.learn(total_timesteps=500_000, callback=live_callback, reset_num_timesteps=False)
    except Exception as e:
        print(f"[Error]: {e}")
    finally:
        print("[*] Menyimpan model akhir...")
        model.save(MODEL_PATH)
        print("✅ Model berhasil disimpan. Training selesai!")
        env.close()

if __name__ == "__main__":
    main()
