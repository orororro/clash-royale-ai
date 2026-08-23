# Clash Royale AI Bot (Hierarchical RL + LLM Macro-Strategist)

Proyek ini adalah simulasi bot AI untuk game bergaya Clash Royale yang menggunakan pendekatan hybrid:
1. **Hierarchical Reinforcement Learning (RL)** menggunakan algoritma PPO untuk mengambil keputusan level mikro (kapan dan unit apa yang harus di-deploy).
2. **LLM Macro-Strategist** menggunakan API LLM untuk menentukan strategi makro (Aggressive, Defensive, Balanced) berdasarkan status pertandingan (HP tower, sisa elixir, dll).

## Komponen Utama
- `clash_env.py`: Custom environment berbasis Gymnasium yang mensimulasikan mekanik game (Elixir, HP, Reward berdasarkan mode strategi).
- `coach.py`: Modul LLM Coach yang terhubung ke endpoint OpenAI-compatible untuk menentukan stategi makro secara dinamis dengan mekanisme fallback.
- `train.py`: Script untuk melatih agen PPO. Menggunakan custom callback untuk mengintegrasikan loop hybrid antara PPO dan LLM Coach.
- `evaluate.py`: Script untuk memuat model yang telah dilatih dan menjalankan simulasi 1 pertandingan dengan log di terminal.

## Prasyarat
Instal semua dependensi menggunakan:
```bash
pip3 install -r requirements.txt
```

*Opsional:* Anda bisa mengubah `api_key` dan `endpoint` pada `coach.py` jika ingin menggunakan provider LLM betulan (seperti OpenRouter / 9router). Secara default menggunakan dummy key dan fallback jika API tidak bisa dihubungi.

## Alur Kerja dan Cara Menjalankan

### 1. Training Model
Jalankan perintah berikut untuk melatih agen:
```bash
python3 train.py
```
- Agen akan belajar melalui interaksi dengan environment (`clash_env.py`).
- Setiap 50 steps (bisa diubah di kode), `train.py` akan meminta stategi makro dari `coach.py` (LLM).
- Strategi ini akan memodulasi bagaimana reward diberikan (misal: mode AGGRESSIVE memberi reward lebih tinggi untuk *damage dealt*).
- Model terlatih akan disimpan dalam folder `models/ppo_clash_bot.zip`.

### 2. Evaluasi Simulasi
Setelah training selesai, Anda bisa menjalankan simulasi evaluasi 1 match penuh:
```bash
python3 evaluate.py
```
- Menjalankan loop inference.
- Secara periodik (tiap 20 langkah), agen akan bertanya pada LLM Coach untuk strategi terbaru.
- Menampilkan visualisasi berbasis teks per langkah dan status pertandingan.
