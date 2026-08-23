import os
import time
from stable_baselines3 import PPO
from arena_env import ArenaEnv
from coach import LLMCoach

def play():
    model_path = "clash_2d_ppo.zip"
    
    if not os.path.exists(model_path) and not os.path.exists("clash_2d_ppo"):
        print(f"Error: Model '{model_path}' tidak ditemukan!")
        print("Jalankan 'python3 train_2d.py' terlebih dahulu untuk melatih bot.")
        return
        
    print("=== Memulai Match Simulasi Visual (AI vs Built-in Enemy) ===")
    
    # 1. Inisialisasi Environment dengan mode render Pygame
    env = ArenaEnv(render_mode="human")
    coach = LLMCoach(model_name="gemini-3.1-flash-lite")
    
    # 2. Muat model terlatih
    model = PPO.load("clash_2d_ppo")
    print("Model PPO berhasil dimuat.")
    
    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0
    total_reward = 0.0
    
    card_names = {0: "Wait", 1: "Knight", 2: "Archer"}
    
    try:
        while not (done or truncated):
            step_count += 1
            
            # Tanya strategi ke LLM Coach setiap 30 langkah
            if step_count % 30 == 0:
                player_hp = (env.agent_tower_l.hp + env.agent_tower_r.hp) / 2.0
                enemy_hp = (env.enemy_tower_l.hp + env.enemy_tower_r.hp) / 2.0
                strategy = coach.get_macro_strategy(player_hp, enemy_hp, env.agent_elixir, env.current_step)
                env.set_strategy_mode(strategy)
                print(f"[COACH] Mode Strategi -> {strategy}")
                
            # Prediksi aksi dari model PPO
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            card_type = int(action[0])
            if card_type in [1, 2]:
                print(f"Step {step_count:03d} | Deploy: {card_names[card_type]} di ({action[1]}, {action[2]}) | Elixir: {env.agent_elixir:.1f}")
                
            time.sleep(0.033)  # ~30 FPS untuk animasi visual yang mulus
            
    except KeyboardInterrupt:
        print("\nMatch dihentikan oleh pengguna.")
    finally:
        print("\n=== Match Selesai ===")
        agent_towers = sum(1 for t in [env.agent_tower_l, env.agent_tower_r] if t.is_alive)
        enemy_towers = sum(1 for t in [env.enemy_tower_l, env.enemy_tower_r] if t.is_alive)
        
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Tower Tersisa - Agen: {agent_towers} | Musuh: {enemy_towers}")
        
        if enemy_towers == 0 and agent_towers > 0:
            print("Hasil Akhir: >>> KEMENANGAN TELAK (VICTORY)! <<<")
        elif agent_towers == 0 and enemy_towers > 0:
            print("Hasil Akhir: >>> KEKALAHAN (DEFEAT)! <<<")
        elif enemy_towers < agent_towers:
            print("Hasil Akhir: >>> MENANG BERDASARKAN SISA TOWER! <<<")
        elif agent_towers < enemy_towers:
            print("Hasil Akhir: >>> KALAH BERDASARKAN SISA TOWER! <<<")
        else:
            print("Hasil Akhir: >>> SERI (DRAW)! <<<")
            
        env.close()

if __name__ == "__main__":
    play()
