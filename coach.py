import os
import requests
import json

class LLMCoach:
    def __init__(self, api_key=None, base_url="http://localhost:20128/v1", model_name="gemini-3.1-flash-lite", model=None):
        # Gunakan API key dari argumen atau environment variable
        self.api_key = api_key or os.environ.get("ROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "9router"
        self.base_url = base_url
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model or model_name
        
    def get_macro_strategy(self, player_hp, enemy_hp, elixir, current_step):
        """Alias untuk get_strategy."""
        return self.get_strategy(player_hp, enemy_hp, elixir, current_step)

    def get_strategy(self, player_hp, enemy_hp, elixir, current_step):
        """
        Mengirim ringkasan match dan mengembalikan strategi makro.
        Returns: 'AGGRESSIVE', 'DEFENSIVE', atau 'BALANCED'.
        """
        prompt = f"""
        Kamu adalah pelatih makro-strategi untuk AI bot Clash Royale.
        Status pertandingan saat ini:
        - Waktu/Step: {current_step}/200
        - HP Tower Pemain: {player_hp}/3000
        - HP Tower Musuh: {enemy_hp}/3000
        - Elixir Pemain: {elixir}/10
        
        Pilih strategi terbaik saat ini:
        - AGGRESSIVE: Fokus menyerang jika HP musuh rendah atau Elixir penuh.
        - DEFENSIVE: Fokus bertahan jika HP pemain rendah atau tertinggal.
        - BALANCED: Seimbang antara menyerang dan bertahan.
        
        Berikan jawaban HANYA dengan satu kata: AGGRESSIVE, DEFENSIVE, atau BALANCED.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        try:
            # Sediakan fallback mechanism jika timeout/gagal
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content'].strip().upper()
                
                if "AGGRESSIVE" in reply:
                    return "AGGRESSIVE"
                elif "DEFENSIVE" in reply:
                    return "DEFENSIVE"
                else:
                    return "BALANCED"
            else:
                print(f"[Coach] API Error ({response.status_code}), fallback to BALANCED.")
                return "BALANCED"
                
        except requests.exceptions.RequestException as e:
            print(f"[Coach] Connection error: {e}, fallback to BALANCED.")
            return "BALANCED"
