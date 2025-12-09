# query_sqlite.py
import sqlite3

DB_PATH = "ecom.db"  # or "ecom.db" depending on what you used

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("▶ Top 5 gaming monitors under $300 (with brand, category, rating):")
    cur.execute("""
        SELECT
            p.product_id,
            p.name AS product_name,
            b.name AS brand_name,
            c.name AS category_name,
            p.price,
            p.avg_rating,
            p.rating_count
        FROM products p
        JOIN brands b
            ON p.brand_id = b.brand_id
        JOIN categories c
            ON p.category_id = c.category_id
        WHERE
            -- adjust this if you dropped search_keywords in your schema
            (p.name LIKE '%gaming%' OR p.short_description LIKE '%gaming%')
            AND p.price < 300
        ORDER BY
            p.avg_rating DESC,
            p.rating_count DESC
        LIMIT 5;
    """)

    rows = cur.fetchall()
    for row in rows:
        (
            product_id,
            product_name,
            brand_name,
            category_name,
            price,
            avg_rating,
            rating_count,
        ) = row
        print(f"- [{product_id}] {product_name} ({brand_name}, {category_name}) "
              f"- ${price:.2f}, rating {avg_rating:.2f} ({rating_count} reviews)")

    print("\n▶ Example: average rating per category (join products + categories):")
    cur.execute("""
        SELECT
            c.name AS category_name,
            COUNT(p.product_id) AS num_products,
            AVG(p.avg_rating) AS avg_category_rating
        FROM categories c
        JOIN products p
            ON p.category_id = c.category_id
        GROUP BY c.category_id
        HAVING num_products >= 3
        ORDER BY avg_category_rating DESC
        LIMIT 10;
    """)

    for row in cur.fetchall():
        category_name, num_products, avg_cat_rating = row
        print(f"- {category_name}: {num_products} products, avg rating {avg_cat_rating:.2f}")

    print("\n▶ Example: top 5 products by total reviews (join products + product_reviews):")
    cur.execute("""
        SELECT
            p.product_id,
            p.name,
            COUNT(r.review_id) AS num_reviews,
            AVG(r.rating) AS avg_review_rating
        FROM products p
        JOIN product_reviews r
            ON p.product_id = r.product_id
        GROUP BY p.product_id
        ORDER BY num_reviews DESC
        LIMIT 5;
    """)

    for row in cur.fetchall():
        pid, name, num_reviews, avg_review_rating = row
        print(f"- [{pid}] {name}: {num_reviews} reviews, avg {avg_review_rating:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
