Saya ingin membuat proyek bot AI untuk game Clash Royale berbasis Hierarchical Reinforcement Learning (RL) + LLM Macro-Strategist menggunakan Python.

Tolong rancang dan buatkan struktur proyek modular lengkap dengan komponen-komponen berikut:

Simulator Lingkungan Game Mini (clash_env.py):

Buat custom environment berbasis Gymnasium (gym.Env).

State/Observation Space (Box): Elixir pemain (0–10), HP Tower pemain, HP Tower musuh, dan Current Strategy Mode (0: Balanced, 1: Aggressive, 2: Defensive).

Action Space (Discrete): 0: Wait, 1: Cheap Unit (Cost 2), 2: Heavy Push (Cost 5).

Modulasi reward function dan kalkulasi damage berdasarkan Strategy Mode yang sedang aktif.

Modul LLM Macro-Strategist (coach.py):

Buat class LLMCoach yang terhubung ke endpoint OpenAI-compatible (seperti 9router / OpenRouter) menggunakan library requests.

Mengirim ringkasan match (HP tower, sisa elixir, riwayat deployment) dan menghasilkan keputusan strategi berupa JSON/teks mode: AGGRESSIVE, DEFENSIVE, atau BALANCED.

Sediakan fallback mechanism jika koneksi API router timeout atau gagal.

Pipeline Training Terintegrasi (train.py):

Gunakan algoritma PPO dari stable-baselines3.

Buat loop hybrid berulang: LLM menentukan strategi makro -> perbarui environment -> jalankan step training PPO -> simpan model checkpoint.

Script Evaluasi / Inference (evaluate.py):

Load model terlatih (.zip) dan jalankan 1 match simulasi dengan visualisasi teks log di terminal.

Dokumentasi & Dependensi:

Buatkan file requirements.txt (gymnasium, stable-baselines3, torch, requests, numpy).

Buatkan file README.md singkat yang menjelaskan alur kerja sistem dan cara menjalankannya.

Silakan mulai dengan membuat file requirements.txt, menginstalnya di terminal, dan menyusun kode file per file secara terstruktur.