"""
Synthetic RL E-commerce Dataset Generator (LLM-assisted)

Strategy:
1. Create primary keys in code.
2. For each table, build a prompt that:
   - Provides allowed IDs / values
   - Asks the LLM to choose from those only
   - Returns valid JSON rows
3. Parse JSON and (optionally) insert into your DB.
"""

import os
import json
import random
from typing import List, Dict, Any

from openai import OpenAI
from tqdm import tqdm  # progress bars

OUTPUT_DIR = "outputs"  # or whatever you like

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_table(name: str, tables: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Save a single table to JSON as soon as it's (partially or fully) generated.
    """
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(tables[name], f, indent=2)
    print(f"💾 Saved {name}: {len(tables[name])} rows -> {path}")


# ------------------------------
# LLM CALL
# ------------------------------

# initialize once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str) -> dict:
    """
    Calls GPT-4o-mini via the OpenAI API to generate JSON data for a given prompt.
    Uses JSON mode and a simple retry loop if parsing fails.
    """
    for attempt in range(3):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},  # 🔒 force valid JSON
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data generator that strictly outputs valid JSON "
                        "matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,  # give it more room so it doesn't truncate
        )

        text = response.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decode failed on attempt {attempt+1}: {e}")
            # fall through to retry

    # If all attempts fail:
    raise RuntimeError("LLM failed to return valid JSON after 3 attempts.")



# ------------------------------
# UTILS
# ------------------------------

def chunk(lst: List[int], n: int) -> List[List[int]]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ------------------------------
# PROMPT BUILDERS
# ------------------------------

def build_categories_prompt(category_ids: List[int]) -> str:
    return f"""
You are generating synthetic rows for an e-commerce `categories` table.

You are given the list of category_ids to populate:
category_ids = {category_ids}

Task:
- For EACH category_id, create exactly ONE category row.
- Create a realistic category hierarchy with both top-level and subcategories.
- Some categories may have parent_category_id = null (top-level).
- Other categories must choose a parent_category_id from the SAME list.

Schema for each row:
- category_id: one of the provided IDs (each used exactly once).
- name: short category name (e.g., "Electronics", "Monitors", "Running Shoes").
- parent_category_id: null for top-level, or another category_id from the list.
- slug: URL-friendly version of name (lowercase, hyphen-separated).

Output ONLY valid JSON:

{{
  "rows": [
    {{
      "category_id": <int>,
      "name": "<string>",
      "parent_category_id": <int or null>,
      "slug": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_brands_prompt(brand_ids: List[int]) -> str:
    return f"""
You are generating synthetic rows for the `brands` table.

brand_ids = {brand_ids}

Task:
- For EACH brand_id, create one brand.
- Use realistic brand names across electronics, beauty, fashion, and home.

Schema per row:
- brand_id: one of the provided IDs (use each exactly once).
- name: short brand name (e.g., "AuroraTech", "Northwind").
- country: country of origin (e.g., "US", "KR", "JP", "DE").

Output JSON only:

{{
  "rows": [
    {{
      "brand_id": <int>,
      "name": "<string>",
      "country": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_users_prompt(user_ids: List[int]) -> str:
    return f"""
Generate synthetic `users` rows.

user_ids = {user_ids}

Task:
- For EACH user_id, create one user.
- Users should be spread across segments (electronics-heavy, beauty-focused, etc.).

Schema:
- user_id: one of the provided IDs.
- signup_channel: one of ["web", "mobile", "referral", "email"].
- segment: short label like "electronics_power_user", "beauty_shopper".
- created_at: ISO 8601 timestamp within the last 2 years.

Output JSON:

{{
  "rows": [
    {{
      "user_id": <int>,
      "signup_channel": "<string>",
      "segment": "<string>",
      "created_at": "<ISO8601>"
    }},
    ...
  ]
}}
""".strip()


def build_attribute_definitions_prompt(attribute_ids: List[int],
                                       category_ids: List[int]) -> str:
    return f"""
Generate synthetic `product_attribute_definitions` rows.

attribute_ids = {attribute_ids}
available_category_ids = {category_ids}

Task:
- Create one row per attribute_id.
- Assign each attribute_id to a category_id from the list.
- Attributes should be useful for filters/compare (e.g., screen_size_inches, skin_type).

Schema:
- attribute_id: one of the provided IDs.
- category_id: choose from available_category_ids.
- name: human-readable name (e.g., "Screen Size").
- data_type: one of ["numeric", "text", "boolean", "enum"].

Output JSON:

{{
  "rows": [
    {{
      "attribute_id": <int>,
      "category_id": <int>,
      "name": "<string>",
      "data_type": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_products_prompt(product_ids: List[int],
                          category_ids: List[int],
                          brand_ids: List[int]) -> str:
    return f"""
Generate synthetic `products` rows.

product_ids = {product_ids}
available_category_ids = {category_ids}
available_brand_ids = {brand_ids}

Task:
- For EACH product_id, create one product.
- Choose a realistic category_id and brand_id from the lists.
- Vary products across monitors, laptops, moisturizers, shoes, coats, coffee makers, etc.

Schema:
- product_id: one of the provided IDs.
- sku: string SKU (e.g., "MON-001-27QHD").
- name: short product name.
- brand_id: from available_brand_ids.
- category_id: from available_category_ids.
- short_description: 1–2 sentences.
- price: numeric price in USD.
- currency: "USD".
- avg_rating: float 1.0–5.0.
- rating_count: int >= 0.
- review_count: int >= 0 (usually <= rating_count).
- search_keywords: string with comma-separated keywords.
- is_active: boolean.

Output JSON:

{{
  "rows": [
    {{
      "product_id": <int>,
      "sku": "<string>",
      "name": "<string>",
      "brand_id": <int>,
      "category_id": <int>,
      "short_description": "<string>",
      "price": <number>,
      "currency": "USD",
      "avg_rating": <number>,
      "rating_count": <int>,
      "review_count": <int>,
      "search_keywords": "<string>",
      "is_active": <bool>
    }},
    ...
  ]
}}
""".strip()


def build_product_variants_prompt(variant_ids: List[int],
                                  product_ids: List[int]) -> str:
    return f"""
Generate `product_variants` rows.

variant_ids = {variant_ids}
available_product_ids = {product_ids}

Task:
- For EACH variant_id, create one variant.
- Each variant must reference a product_id from the list.
- Vary colors and size labels realistically.

Schema:
- variant_id: one of the provided IDs.
- product_id: from available_product_ids.
- variant_sku: string (e.g., "MON-001-27QHD-BLK").
- color_name: e.g., "Black", "Navy", "White".
- size_label: e.g., "8.5", "M", "27-inch", etc.
- additional_price_delta: numeric (can be negative or positive).
- stock_status: one of ["in_stock", "low_stock", "out_of_stock"].

Output JSON:

{{
  "rows": [
    {{
      "variant_id": <int>,
      "product_id": <int>,
      "variant_sku": "<string>",
      "color_name": "<string>",
      "size_label": "<string>",
      "additional_price_delta": <number>,
      "stock_status": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_product_images_prompt(image_ids: List[int],
                                product_ids: List[int]) -> str:
    return f"""
Generate `product_images` rows.

image_ids = {image_ids}
available_product_ids = {product_ids}

Task:
- For EACH image_id, create one image.
- Each image references a product_id.
- Some products should have multiple images; at least one image per product_id across the dataset.

Schema:
- image_id: one of the provided IDs.
- product_id: from available_product_ids.
- url: image URL string.
- alt_text: short description of the image.
- is_primary: boolean.
- display_order: integer >= 1.

Output JSON:

{{
  "rows": [
    {{
      "image_id": <int>,
      "product_id": <int>,
      "url": "<string>",
      "alt_text": "<string>",
      "is_primary": <bool>,
      "display_order": <int>
    }},
    ...
  ]
}}
""".strip()


def build_attribute_values_prompt(product_ids: List[int],
                                  attribute_ids: List[int]) -> str:
    return f"""
Generate `product_attribute_values` rows.

available_product_ids = {product_ids}
available_attribute_ids = {attribute_ids}

Task:
- Create rows that assign attribute values to products.
- For each row, choose product_id and attribute_id from the lists.
- Choose realistic values based on typical e-commerce attributes
  (screen sizes, resolutions, skin types, fill_power, etc.).
- It is OK if not every (product, attribute) pair appears, but there should be broad coverage.

Schema:
- product_id: from available_product_ids.
- attribute_id: from available_attribute_ids.
- value: string representation; numeric attributes should be numeric-looking strings.

Output JSON:

{{
  "rows": [
    {{
      "product_id": <int>,
      "attribute_id": <int>,
      "value": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_product_reviews_prompt(review_ids: List[int],
                                 products_subset: List[Dict[str, Any]],
                                 user_ids: List[int]) -> str:
    """
    products_subset is a list of dicts like:
    { "product_id": 1, "name": "AuroraView 27\" QHD 144Hz Monitor" }
    """
    product_examples = json.dumps(products_subset, indent=2)
    available_product_ids = [p["product_id"] for p in products_subset]

    return f"""
Generate `product_reviews` rows.

review_ids = {review_ids}
available_product_ids = {available_product_ids}
available_user_ids = {user_ids}

You MUST only use product_id values from the list: {available_product_ids}.

Products context (id + name):
{product_examples}

Task:
- For EACH review_id, create one review.
- Reference product_id and user_id from the lists.
- Ratings should be mostly positive but with some low ratings.
- Review bodies should clearly refer to the correct product (by its name/category).

Schema:
- review_id: one of the provided IDs.
- product_id: from available_product_ids.
- user_id: from available_user_ids.
- rating: integer 1–5.
- title: short title.
- body: 1-3 sentence review text.
- created_at: ISO timestamp.
- verified_purchase: boolean.
- helpful_vote_count: integer >= 0.
- user_segment: short descriptor (e.g., "power_user", "casual_shopper").

Output JSON:

{{
  "rows": [
    {{
      "review_id": <int>,
      "product_id": <int>,
      "user_id": <int>,
      "rating": <int>,
      "title": "<string>",
      "body": "<string>",
      "created_at": "<ISO8601>",
      "verified_purchase": <bool>,
      "helpful_vote_count": <int>,
      "user_segment": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_sessions_prompt(session_ids: List[int],
                          user_ids: List[int]) -> str:
    return f"""
Generate `sessions` rows.

session_ids = {session_ids}
available_user_ids = {user_ids}

Task:
- For EACH session_id, create one session.
- Each session belongs to a user_id from the list.
- journey_type is either "search_purchase" or "save_purchase".
- started_at < ended_at.

Schema:
- session_id: from session_ids.
- user_id: from available_user_ids.
- journey_type: "search_purchase" or "save_purchase".
- started_at: ISO timestamp.
- ended_at: ISO timestamp (later than started_at).
- entry_point: e.g., "home", "saved_list", "email".
- device_type: one of ["desktop", "mobile"].

Output JSON:

{{
  "rows": [
    {{
      "session_id": <int>,
      "user_id": <int>,
      "journey_type": "<string>",
      "started_at": "<ISO8601>",
      "ended_at": "<ISO8601>",
      "entry_point": "<string>",
      "device_type": "<string>"
    }},
    ...
  ]
}}
""".strip()


def build_events_prompt(event_ids: List[int],
                        session_ids: List[int],
                        user_ids: List[int],
                        product_ids: List[int]) -> str:
    return f"""
Generate `events` rows.

event_ids = {event_ids}
available_session_ids = {session_ids}
available_user_ids = {user_ids}
available_product_ids = {product_ids}

Allowed event_type values:
- "search", "apply_filter", "view_product",
  "view_reviews", "search_reviews",
  "save_product", "open_saved_list", "purchase_product"

Task:
- Create one event row per event_id.
- Each event references an existing session_id (and its user_id).
- For product-related events, choose product_id from available_product_ids.
- Ensure that within each session, step_index is roughly increasing and
  there is at most one "purchase_product" event.

Schema:
- event_id: from event_ids.
- session_id: from available_session_ids.
- user_id: from available_user_ids.
- step_index: integer >= 0.
- event_type: from the allowed list.
- query_text: non-empty only for "search".
- filters_json: JSON string describing filters, may be empty.
- product_id: null or from available_product_ids for product events.
- position_in_list: integer position or null.
- timestamp: ISO timestamp.
- is_terminal: boolean (true only for final purchase event in a session).

Output JSON:

{{
  "rows": [
    {{
      "event_id": <int>,
      "session_id": <int>,
      "user_id": <int>,
      "step_index": <int>,
      "event_type": "<string>",
      "query_text": "<string or empty>",
      "filters_json": "<JSON string>",
      "product_id": <int or null>,
      "position_in_list": <int or null>,
      "timestamp": "<ISO8601>",
      "is_terminal": <bool>
    }},
    ...
  ]
}}
""".strip()


def build_saved_list_prompt(saved_list_ids: List[int],
                            user_ids: List[int],
                            event_ids: List[int]) -> str:
    return f"""
Generate `saved_list` rows.

saved_list_ids = {saved_list_ids}
available_user_ids = {user_ids}
available_event_ids = {event_ids}

Task:
- For EACH saved_list_id, create one list.
- Each list belongs to a user_id.
- event_id should reference an event where the user saved an item
  (you may choose any event_id from the list).

Schema:
- saved_list_id: from saved_list_ids.
- event_id: from available_event_ids.
- user_id: from available_user_ids.
- name: short list name (e.g., "Half-marathon shoes", "Work monitors").
- created_at: ISO timestamp.

Output JSON:

{{
  "rows": [
    {{
      "saved_list_id": <int>,
      "event_id": <int>,
      "user_id": <int>,
      "name": "<string>",
      "created_at": "<ISO8601>"
    }},
    ...
  ]
}}
""".strip()


def build_saved_list_items_prompt(saved_list_item_ids: List[int],
                                  saved_list_ids: List[int],
                                  product_ids: List[int],
                                  session_ids: List[int],
                                  event_ids: List[int]) -> str:
    return f"""
Generate `saved_list_items` rows.

saved_list_item_ids = {saved_list_item_ids}
available_saved_list_ids = {saved_list_ids}
available_product_ids = {product_ids}
available_session_ids = {session_ids}
available_event_ids = {event_ids}

Task:
- For EACH saved_list_item_id, create one row.
- Each row ties a saved_list_id to a product_id.
- source_session_id and source_event_id should correspond to where the save happened.

Schema:
- saved_list_item_id: from saved_list_item_ids.
- saved_list_id: from available_saved_list_ids.
- product_id: from available_product_ids.
- saved_at: ISO timestamp.
- source_session_id: from available_session_ids.
- source_event_id: from available_event_ids.

Output JSON:

{{
  "rows": [
    {{
      "saved_list_item_id": <int>,
      "saved_list_id": <int>,
      "product_id": <int>,
      "saved_at": "<ISO8601>",
      "source_session_id": <int>,
      "source_event_id": <int>
    }},
    ...
  ]
}}
""".strip()


def build_purchases_prompt(purchase_ids: List[int],
                           event_ids: List[int],
                           session_ids: List[int],
                           user_ids: List[int],
                           product_ids: List[int],
                           variant_ids: List[int]) -> str:
    return f"""
Generate `purchases` rows.

purchase_ids = {purchase_ids}
available_event_ids = {event_ids}
available_session_ids = {session_ids}
available_user_ids = {user_ids}
available_product_ids = {product_ids}
available_variant_ids = {variant_ids}

Task:
- For EACH purchase_id, create one purchase.
- event_id should correspond to an event with event_type="purchase_product".
- session_id and user_id should be consistent with that event.
- product_id and variant_id must come from the provided lists.

Schema:
- purchase_id: from purchase_ids.
- event_id: from available_event_ids.
- session_id: from available_session_ids.
- user_id: from available_user_ids.
- product_id: from available_product_ids.
- variant_id: from available_variant_ids (or null if not applicable).
- quantity: integer >= 1.
- price_paid: numeric, roughly matching product price.
- purchased_at: ISO timestamp.
- source_journey_type: "search_purchase" or "save_purchase".

Output JSON:

{{
  "rows": [
    {{
      "purchase_id": <int>,
      "event_id": <int>,
      "session_id": <int>,
      "user_id": <int>,
      "product_id": <int>,
      "variant_id": <int or null>,
      "quantity": <int>,
      "price_paid": <number>,
      "purchased_at": "<ISO8601>",
      "source_journey_type": "<string>"
    }},
    ...
  ]
}}
""".strip()


# ------------------------------
# MAIN GENERATION PIPELINE
# ------------------------------

def main():
    NUM_ROWS = 100
    BATCH_SIZE = 25  # how many rows to ask for per prompt

    # 1) ID creation
    category_ids        = list(range(1, NUM_ROWS + 1))
    brand_ids           = list(range(1, NUM_ROWS + 1))
    user_ids            = list(range(1, NUM_ROWS + 1))
    attribute_ids       = list(range(1, NUM_ROWS + 1))

    product_ids         = list(range(1, NUM_ROWS + 1))
    variant_ids         = list(range(1, NUM_ROWS + 1))
    image_ids           = list(range(1, NUM_ROWS + 1))
    review_ids          = list(range(1, NUM_ROWS + 1))

    session_ids         = list(range(1, NUM_ROWS + 1))
    event_ids           = list(range(1, NUM_ROWS + 1))
    purchase_ids        = list(range(1, NUM_ROWS + 1))
    saved_list_ids      = list(range(1, NUM_ROWS + 1))
    saved_list_item_ids = list(range(1, NUM_ROWS + 1))

    tables: Dict[str, List[Dict[str, Any]]] = {
        "categories": [],
        "brands": [],
        "users": [],
        "product_attribute_definitions": [],
        "products": [],
        "product_variants": [],
        "product_images": [],
        "product_attribute_values": [],
        "product_reviews": [],
        "sessions": [],
        "events": [],
        "saved_list": [],
        "saved_list_items": [],
        "purchases": [],
    }

    print("🚀 Starting synthetic dataset generation...")

    # # 2) Generate categories
    # for chunk_ids in tqdm(list(chunk(category_ids, BATCH_SIZE)), desc="Categories"):
    #     prompt = build_categories_prompt(chunk_ids)
    #     resp = call_llm(prompt)
    #     tables["categories"].extend(resp["rows"])
    # save_table("categories", tables)

    # # 3) Generate brands
    # for chunk_ids in tqdm(list(chunk(brand_ids, BATCH_SIZE)), desc="Brands"):
    #     prompt = build_brands_prompt(chunk_ids)
    #     resp = call_llm(prompt)
    #     tables["brands"].extend(resp["rows"])
    # save_table("brands", tables)

    # # 4) Generate users
    # for chunk_ids in tqdm(list(chunk(user_ids, BATCH_SIZE)), desc="Users"):
    #     prompt = build_users_prompt(chunk_ids)
    #     resp = call_llm(prompt)
    #     tables["users"].extend(resp["rows"])
    # save_table("users", tables)

    # # 5) Attribute definitions
    # for chunk_ids in tqdm(list(chunk(attribute_ids, BATCH_SIZE)), desc="Attr defs"):
    #     prompt = build_attribute_definitions_prompt(chunk_ids, category_ids)
    #     resp = call_llm(prompt)
    #     tables["product_attribute_definitions"].extend(resp["rows"])
    # save_table("product_attribute_definitions", tables)
    # 6) Products
    for chunk_ids in tqdm(list(chunk(product_ids, BATCH_SIZE)), desc="Products"):
        prompt = build_products_prompt(chunk_ids, category_ids, brand_ids)
        resp = call_llm(prompt)
        tables["products"].extend(resp["rows"])
    save_table("products", tables)

    # 7) Product variants
    for chunk_ids in tqdm(list(chunk(variant_ids, BATCH_SIZE)), desc="Variants"):
        prompt = build_product_variants_prompt(chunk_ids, product_ids)
        resp = call_llm(prompt)
        tables["product_variants"].extend(resp["rows"])
    save_table("product_variants", tables)

    # 8) Product images
    for chunk_ids in tqdm(list(chunk(image_ids, BATCH_SIZE)), desc="Images"):
        prompt = build_product_images_prompt(chunk_ids, product_ids)
        resp = call_llm(prompt)
        tables["product_images"].extend(resp["rows"])
    save_table("product_images", tables)

    # 9) Product attribute values
    print("Generating attribute values...")
    prompt = build_attribute_values_prompt(product_ids, attribute_ids)
    resp = call_llm(prompt)
    tables["product_attribute_values"].extend(resp["rows"])
    save_table("product_attribute_values", tables)

    # Build mapping product_id -> name for reviews
    product_meta = [
        {"product_id": p["product_id"], "name": p["name"]}
        for p in tables["products"]
    ]
    random.shuffle(product_meta)

    # 10) Product reviews
    for chunk_ids in tqdm(list(chunk(review_ids, BATCH_SIZE)), desc="Reviews"):
        subset = random.sample(product_meta, min(len(product_meta), 10))
        prompt = build_product_reviews_prompt(chunk_ids, subset, user_ids)
        resp = call_llm(prompt)
        tables["product_reviews"].extend(resp["rows"])
    save_table("product_reviews", tables)

    # 11) Sessions
    for chunk_ids in tqdm(list(chunk(session_ids, BATCH_SIZE)), desc="Sessions"):
        prompt = build_sessions_prompt(chunk_ids, user_ids)
        resp = call_llm(prompt)
        tables["sessions"].extend(resp["rows"])
    save_table("sessions", tables)

    # 12) Events
    for chunk_ids in tqdm(list(chunk(event_ids, BATCH_SIZE)), desc="Events"):
        prompt = build_events_prompt(chunk_ids, session_ids, user_ids, product_ids)
        resp = call_llm(prompt)
        tables["events"].extend(resp["rows"])
    save_table("events", tables)

    # 13) Saved lists
    for chunk_ids in tqdm(list(chunk(saved_list_ids, BATCH_SIZE)), desc="Saved lists"):
        prompt = build_saved_list_prompt(chunk_ids, user_ids, event_ids)
        resp = call_llm(prompt)
        tables["saved_list"].extend(resp["rows"])
    save_table("saved_list", tables)

    # 14) Saved list items
    for chunk_ids in tqdm(list(chunk(saved_list_item_ids, BATCH_SIZE)), desc="Saved items"):
        prompt = build_saved_list_items_prompt(
            chunk_ids, saved_list_ids, product_ids, session_ids, event_ids
        )
        resp = call_llm(prompt)
        tables["saved_list_items"].extend(resp["rows"])
    save_table("saved_list_items", tables)

    # 15) Purchases
    for chunk_ids in tqdm(list(chunk(purchase_ids, BATCH_SIZE)), desc="Purchases"):
        prompt = build_purchases_prompt(
            chunk_ids, event_ids, session_ids, user_ids, product_ids, variant_ids
        )
        resp = call_llm(prompt)
        tables["purchases"].extend(resp["rows"])
    save_table("purchases", tables)

    print("✅ Generation complete.")
    print(f"Products: {len(tables['products'])}, Reviews: {len(tables['product_reviews'])}, "
          f"Sessions: {len(tables['sessions'])}, Events: {len(tables['events'])}")

    print("Example: first 2 products:")
    print(json.dumps(tables["products"][:2], indent=2))


if __name__ == "__main__":
    main()
