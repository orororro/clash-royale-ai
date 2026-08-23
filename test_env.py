import time
import random
from arena_env import ArenaEnv

def test_visual_arena():
    print("=== Memulai Uji Coba Simulator Arena 2D (Pygame) ===")
    env = ArenaEnv(render_mode="human")
    obs, info = env.reset()
    
    done = False
    truncated = False
    step_count = 0
    
    # Mode switch testing
    modes = ["BALANCED", "AGGRESSIVE", "DEFENSIVE"]
    
    try:
        while not (done or truncated) and step_count < 200:
            step_count += 1
            
            # Ganti mode strategi setiap 40 langkah untuk uji coba
            if step_count % 40 == 0:
                new_mode = modes[(step_count // 40) % len(modes)]
                env.set_strategy_mode(new_mode)
                print(f"[TEST] Berganti ke Mode Strategi: {new_mode}")
                
            # Contoh aksi agen: Menaruh Knight / Archer secara berkala di sisi bawah (y >= 16)
            if env.agent_elixir >= 3.0 and random.random() < 0.3:
                card_type = random.choice([1, 2])  # 1: Knight, 2: Archer
                pos_x = random.choice([3, 4, 13, 14])  # Dekat jembatan kiri atau kanan
                pos_y = random.randint(18, 26)  # Di wilayah agen
                action = [card_type, pos_x, pos_y]
            else:
                action = [0, 0, 0]  # Wait
                
            obs, reward, done, truncated, info = env.step(action)
            
            if step_count % 20 == 0:
                print(f"Step {step_count:03d} | Agent Elixir: {env.agent_elixir:.1f} | Units di arena: {len(env.units)} | Reward: {reward:.2f}")
                
            time.sleep(0.03)  # Kecepatan animasi visual
            
    except KeyboardInterrupt:
        print("\nUji coba dihentikan oleh pengguna.")
    finally:
        env.close()
        print("=== Uji Coba Selesai ===")

if __name__ == "__main__":
    test_visual_arena()
