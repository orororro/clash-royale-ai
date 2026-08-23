import os
import signal
import sys
import time
from stable_baselines3 import PPO
from arena_env import ArenaEnv  # Diperbaiki dari ClashArenaEnv agar sesuai dengan class yang ada
from coach import LLMCoach

# 1. Inisialisasi Coach Hermes
coach = LLMCoach(
    api_key="sk-1d47fd2534d0197b-8whljv-cfeeba41",
    base_url="http://localhost:20128/v1",
    model_name="gemini-3.1-flash-lite"
)

# 2. Inisialisasi Environment
env = ArenaEnv(render_mode=None)

# 3. Load Model Lama jika ada, atau buat baru
MODEL_PATH = "clash_2d_ppo.zip"
if os.path.exists(MODEL_PATH):
    print(f"[*] Melanjutkan training dari {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("[*] Membuat model PPO baru...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 4. Handler untuk Safe Shutdown saat ditekan CTRL+C
stop_requested = False
def signal_handler(sig, frame):
    global stop_requested
    print("\n[!] Perintah STOP diterima! Menyimpan model sebelum keluar...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)

print("\n" + "="*50)
print("🚀 TRAINING DIMULAI (Tekan Ctrl + C di terminal untuk STOP & SAVE)")
print("="*50 + "\n")

iteration = 1
STEPS_PER_CYCLE = 10_000  # Setiap 10k step, minta review taktik ke Hermes

try:
    while not stop_requested:
        print(f"\n--- [Cycle #{iteration}] Menjalankan {STEPS_PER_CYCLE} Timesteps ---")
        
        # Training sejumlah step
        model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        
        # Minta arahan strategi makro ke Hermes Coach
        summary = {
            "my_tower_hp": 850,     # Bisa diambil dari metrik env rata-rata
            "enemy_tower_hp": 600,
            "avg_elixir": 6.5
        }
        
        # Memanggil get_macro_strategy dengan parameter yang benar (player_hp, enemy_hp, elixir, current_step)
        strategy = coach.get_macro_strategy(
            summary["my_tower_hp"], 
            summary["enemy_tower_hp"], 
            summary["avg_elixir"], 
            iteration * STEPS_PER_CYCLE
        )
        print(f"🤖 [Hermes Coach Review]: Mode Taktik Terpilih -> {strategy}")
        
        # Simpan checkpoint otomatis tiap siklus
        model.save("clash_2d_ppo")
        print(f"💾 Checkpoint tersimpan di clash_2d_ppo.zip")
        
        iteration += 1

except Exception as e:
    print(f"[Error]: {e}")

finally:
    print("[*] Menyimpan model akhir...")
    model.save("clash_2d_ppo")
    print("✅ Model berhasil disimpan. Training selesai!")
    sys.exit(0)
