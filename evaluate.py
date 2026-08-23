from stable_baselines3 import PPO
from clash_env import ClashEnv
from coach import LLMCoach
import time

def evaluate():
    env = ClashEnv()
    coach = LLMCoach()
    
    try:
        model = PPO.load("models/ppo_clash_bot")
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        print("Pastikan anda sudah menjalankan train.py terlebih dahulu.")
        return

    obs, info = env.reset()
    done = False
    truncated = False
    
    print("=== Memulai Match Simulasi ===")
    
    while not (done or truncated):
        # Tanya LLM Coach setiap 20 langkah selama evaluasi
        if env.current_step % 20 == 0:
            elixir = env.state[0]
            player_hp = env.state[1]
            enemy_hp = env.state[2]
            strategy = coach.get_strategy(player_hp, enemy_hp, elixir, env.current_step)
            env.set_strategy_mode(strategy)
            print(f"\n[COACH] Mengubah strategi menjadi: {strategy}\n")
            
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        env.render()
        
        # Log detil interaksi action
        action_map = {0: "Wait", 1: "Cheap Unit (2 Elixir)", 2: "Heavy Push (5 Elixir)"}
        print(f"-> Agent memilih aksi: {action_map[int(action)]} | Reward: {reward:.2f}")
        
        time.sleep(0.1) # Jeda untuk visualisasi teks
        
    print("=== Match Selesai ===")
    if env.state[2] <= 0 and env.state[1] > 0:
        print("Hasil: MENANG!")
    elif env.state[1] <= 0:
        print("Hasil: KALAH!")
    else:
        print("Hasil: SERI / Timeout")

if __name__ == "__main__":
    evaluate()
