from coach import LLMCoach

def test_connection():
    print("=== Testing LLMCoach Connection ===")
    
    coach = LLMCoach(
        api_key="sk-1d47fd2534d0197b-8whljv-cfeeba41",
        base_url="http://localhost:20128/v1",
        model_name="nousresearch/hermes-3-llama-3.1-8b"
    )
    
    # Dummy stats for testing
    player_hp = 2500
    enemy_hp = 1000
    elixir = 10
    current_step = 150
    
    print(f"Mengirim request ke {coach.endpoint} menggunakan model {coach.model}...")
    
    # Memanggil get_macro_strategy yang akan memanggil OpenRouter/9Router compatible endpoint
    strategy = coach.get_macro_strategy(player_hp, enemy_hp, elixir, current_step)
    
    print(f"\n[Hasil LLM Coach]: Strategi yang terpilih adalah -> {strategy}")
    print("=== Test Selesai ===")

if __name__ == "__main__":
    test_connection()
