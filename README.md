# 🛒 Synthetic E-Commerce RL Dataset

This repository generates a **synthetic e-commerce dataset** for **reinforcement learning environments** and **LLM-based user simulation**.

It includes three main scripts:

1. **`generate_data.py`** — creates synthetic, relational e-commerce data using OpenAI's `gpt-4o-mini`.
2. **`load_sqlite.py`** — loads all generated JSON tables into a SQLite database.
3. **`query_sqlite.py`** — runs example multi-table SQL queries and joins to explore the dataset.

---

## 🧠 Overview

The dataset models an end-to-end e-commerce ecosystem:

- **Catalog layer:** categories, brands, products, variants, attributes, and images.  
- **User layer:** users, sessions, and behavioral events.  
- **Interaction layer:** saved lists, reviews, and purchases.  

Each table is generated via **structured JSON prompts** to ensure schema integrity and relational consistency.

---

## 📦 Installation

### 1. Clone the repository and set up the environment
```bash
git clone https://github.com/sunotsue/ecom-env-data.git
cd ecom-env-dataset
python3.10 -m venv ecom
source ecom/bin/activate
pip install openai tqdm
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🚀 Step 1 — Generate Synthetic Data

Run the generator to create all tables:
```bash
python generate_data.py
```

**Features:**

* Uses `gpt-4o-mini` to generate realistic e-commerce data.
* Each table (e.g., `products`, `users`, `events`) contains 100+ rows.
* Progress bars show generation status.
* Auto-saves each table as JSON in the `outputs/` folder.

Example output:
```
🚀 Starting synthetic dataset generation...
Categories: 100%|███████████████████████| 4/4 [01:15<00:00, 18.9s/it]
💾 Saved categories: 100 rows -> outputs/categories.json
✅ Generation complete.
```

---

## 🗄 Step 2 — Load Data into SQLite

Once JSON files are generated, load them into a local SQLite database:
```bash
python load_sqlite.py
```

**What it does:**

* Creates `ecom.db` (or `ecom_rl.db`) in the root directory.
* Builds normalized tables for all entities.
* Inserts data from `outputs/*.json`.

Check your database:
```bash
sqlite3 ecom.db
sqlite> .tables
sqlite> SELECT COUNT(*) FROM products;
```

---

## 🔍 Step 3 — Query the Dataset

Run the query examples to explore the data:
```bash
python query_sqlite.py
```

**Included examples:**

1. **Top gaming products under $300**
   * Joins `products`, `brands`, and `categories`.
2. **Average rating per category**
   * Joins `products` and `categories`.
3. **Most-reviewed products**
   * Joins `products` and `product_reviews`.

**Sample output:**
```
▶ Top 5 gaming monitors under $300 (with brand, category, rating):
- [42] AuroraView 27" QHD (AuroraTech, Monitors) - $249.99, rating 4.7 (213 reviews)

▶ Average rating per category:
- Laptops: 14 products, avg rating 4.65

▶ Top 5 products by total reviews:
- [18] AirRunner Pro Shoes: 132 reviews, avg 4.8
```

---

## 🧩 Project Structure
```
├── generate_data.py          # LLM-assisted data generator
├── load_sqlite.py            # Load JSON → SQLite
├── query_sqlite.py           # Example joins & analytics
├── outputs/                  # Saved JSON tables
├── ecom.db                   # SQLite database (after loading)
└── README.md
```

---

## ⚙️ Customization

* Modify `NUM_ROWS` and `BATCH_SIZE` in `generate_data.py` to scale dataset size.
* Edit `build_*_prompt()` functions to adjust schema or add new domains.
* Replace `gpt-4o-mini` with a larger model (e.g., `gpt-4-turbo`) for richer content.
* Use the saved JSONs directly in Pandas, DuckDB, or BigQuery for analysis.

---

## 🧾 Research Applications

* Reinforcement learning environments for **user interaction simulation**.
* Fine-tuning datasets for **tool-using and reasoning agents**.
* Training **recommendation or ranking models**.
* Generating **SQL reasoning / schema alignment** benchmarks.

---

## 🪶 License

MIT License © 2025 Your Name