import os
import json
import sqlite3
from typing import List, Dict, Any

OUTPUT_DIR = "outputs"
DB_PATH = "ecom.db"


def load_json(table_name: str) -> List[Dict[str, Any]]:
    path = os.path.join(OUTPUT_DIR, f"{table_name}.json")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping.")
        return []
    with open(path, "r") as f:
        return json.load(f)


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --------------------------
    # 1. Drop + create tables
    # --------------------------
    # NOTE: Adjust schemas here if you change the generator.
    ddl_statements = {
        "categories": """
            DROP TABLE IF EXISTS categories;
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                name TEXT,
                parent_category_id INTEGER,
                slug TEXT
            );
        """,
        "brands": """
            DROP TABLE IF EXISTS brands;
            CREATE TABLE brands (
                brand_id INTEGER PRIMARY KEY,
                name TEXT,
                country TEXT
            );
        """,
        "users": """
            DROP TABLE IF EXISTS users;
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                signup_channel TEXT,
                segment TEXT,
                created_at TEXT
            );
        """,
        "product_attribute_definitions": """
            DROP TABLE IF EXISTS product_attribute_definitions;
            CREATE TABLE product_attribute_definitions (
                attribute_id INTEGER PRIMARY KEY,
                category_id INTEGER,
                name TEXT,
                data_type TEXT
            );
        """,
        "products": """
            DROP TABLE IF EXISTS products;
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                sku TEXT,
                name TEXT,
                brand_id INTEGER,
                category_id INTEGER,
                short_description TEXT,
                price REAL,
                currency TEXT,
                avg_rating REAL,
                rating_count INTEGER,
                review_count INTEGER,
                search_keywords TEXT,
                is_active INTEGER
            );
        """,
        "product_variants": """
            DROP TABLE IF EXISTS product_variants;
            CREATE TABLE product_variants (
                variant_id INTEGER PRIMARY KEY,
                product_id INTEGER,
                variant_sku TEXT,
                color_name TEXT,
                size_label TEXT,
                additional_price_delta REAL,
                stock_status TEXT
            );
        """,
        "product_images": """
            DROP TABLE IF EXISTS product_images;
            CREATE TABLE product_images (
                image_id INTEGER PRIMARY KEY,
                product_id INTEGER,
                url TEXT,
                alt_text TEXT,
                is_primary INTEGER,
                display_order INTEGER
            );
        """,
        "product_attribute_values": """
            DROP TABLE IF EXISTS product_attribute_values;
            CREATE TABLE product_attribute_values (
                product_id INTEGER,
                attribute_id INTEGER,
                value TEXT
            );
        """,
        "product_reviews": """
            DROP TABLE IF EXISTS product_reviews;
            CREATE TABLE product_reviews (
                review_id INTEGER PRIMARY KEY,
                product_id INTEGER,
                user_id INTEGER,
                rating INTEGER,
                title TEXT,
                body TEXT,
                created_at TEXT,
                verified_purchase INTEGER,
                helpful_vote_count INTEGER,
                user_segment TEXT
            );
        """,
        "sessions": """
            DROP TABLE IF EXISTS sessions;
            CREATE TABLE sessions (
                session_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                journey_type TEXT,
                started_at TEXT,
                ended_at TEXT,
                entry_point TEXT,
                device_type TEXT
            );
        """,
        "events": """
            DROP TABLE IF EXISTS events;
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY,
                session_id INTEGER,
                user_id INTEGER,
                step_index INTEGER,
                event_type TEXT,
                query_text TEXT,
                filters_json TEXT,
                product_id INTEGER,
                position_in_list INTEGER,
                timestamp TEXT,
                is_terminal INTEGER
            );
        """,
        "saved_list": """
            DROP TABLE IF EXISTS saved_list;
            CREATE TABLE saved_list (
                saved_list_id INTEGER PRIMARY KEY,
                event_id INTEGER,
                user_id INTEGER,
                name TEXT,
                created_at TEXT
            );
        """,
        "saved_list_items": """
            DROP TABLE IF EXISTS saved_list_items;
            CREATE TABLE saved_list_items (
                saved_list_item_id INTEGER PRIMARY KEY,
                saved_list_id INTEGER,
                product_id INTEGER,
                saved_at TEXT,
                source_session_id INTEGER,
                source_event_id INTEGER
            );
        """,
        "purchases": """
            DROP TABLE IF EXISTS purchases;
            CREATE TABLE purchases (
                purchase_id INTEGER PRIMARY KEY,
                event_id INTEGER,
                session_id INTEGER,
                user_id INTEGER,
                product_id INTEGER,
                variant_id INTEGER,
                quantity INTEGER,
                price_paid REAL,
                purchased_at TEXT,
                source_journey_type TEXT
            );
        """,
    }

    for table, ddl in ddl_statements.items():
        print(f"🔧 Creating table: {table}")
        cur.executescript(ddl)

    conn.commit()

    # --------------------------
    # 2. Load JSON + insert rows
    # --------------------------

    def insert_rows(table: str, cols: List[str]):
        rows = load_json(table)
        if not rows:
            print(f"⚠️  No rows for {table}, skipping insert.")
            return

        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

        values = []
        for r in rows:
            values.append([r.get(c) for c in cols])

        cur.executemany(sql, values)
        conn.commit()
        print(f"✅ Inserted {len(rows)} rows into {table}")

    # Columns MUST match schemas above
    insert_rows("categories", [
        "category_id", "name", "parent_category_id", "slug"
    ])

    insert_rows("brands", [
        "brand_id", "name", "country"
    ])

    insert_rows("users", [
        "user_id", "signup_channel", "segment", "created_at"
    ])

    insert_rows("product_attribute_definitions", [
        "attribute_id", "category_id", "name", "data_type"
    ])

    insert_rows("products", [
        "product_id", "sku", "name", "brand_id", "category_id",
        "short_description", "price", "currency",
        "avg_rating", "rating_count", "review_count",
        "search_keywords", "is_active"
    ])

    insert_rows("product_variants", [
        "variant_id", "product_id", "variant_sku",
        "color_name", "size_label", "additional_price_delta", "stock_status"
    ])

    insert_rows("product_images", [
        "image_id", "product_id", "url", "alt_text", "is_primary", "display_order"
    ])

    insert_rows("product_attribute_values", [
        "product_id", "attribute_id", "value"
    ])

    insert_rows("product_reviews", [
        "review_id", "product_id", "user_id", "rating",
        "title", "body", "created_at", "verified_purchase",
        "helpful_vote_count", "user_segment"
    ])

    insert_rows("sessions", [
        "session_id", "user_id", "journey_type",
        "started_at", "ended_at", "entry_point", "device_type"
    ])

    insert_rows("events", [
        "event_id", "session_id", "user_id", "step_index",
        "event_type", "query_text", "filters_json",
        "product_id", "position_in_list", "timestamp", "is_terminal"
    ])

    insert_rows("saved_list", [
        "saved_list_id", "event_id", "user_id", "name", "created_at"
    ])

    insert_rows("saved_list_items", [
        "saved_list_item_id", "saved_list_id", "product_id",
        "saved_at", "source_session_id", "source_event_id"
    ])

    insert_rows("purchases", [
        "purchase_id", "event_id", "session_id", "user_id",
        "product_id", "variant_id", "quantity",
        "price_paid", "purchased_at", "source_journey_type"
    ])

    conn.close()
    print(f"🎉 All done. Data loaded into {DB_PATH}")


if __name__ == "__main__":
    main()
