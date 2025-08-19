import json
import os
from typing import List, Dict, Any

# Import configurations and utility functions
from dotenv import load_dotenv
from schema_utils import load_and_process_schema, find_semantic_candidates, process_batch_with_llm

def main():
    """Main orchestration function."""
    print("--- Starting TextSQL to VectorSQL Expansion Process ---")

    # 1. Load and process schemas
    load_dotenv()
    print(f'''Loading schemas from {os.getenv("SCHEMA_FILE_PATH")}...''')
    schemas = load_and_process_schema(os.getenv("SCHEMA_FILE_PATH"))
    if not schemas:
        return
    print(f"Successfully loaded and processed {len(schemas)} database schemas.")

    # 2. Load input data
    print(f'''Loading input data from {os.getenv("INPUT_FILE_PATH")}...''')
    try:
        with open(os.getenv("INPUT_FILE_PATH"), 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f'''Error: Input file not found at {os.getenv("INPUT_FILE_PATH")}''')
        return
    print(f"Loaded {len(input_data)} question/SQL pairs.")
    
    # 3. Identify convertible items
    items_to_process = []
    unconvertible_items = []

    for item in input_data:
        db_id = item.get("db_id")
        sql = item.get("query")
        schema = schemas.get(db_id)

        if not db_id or not sql or not schema:
            item["CONVERT"] = "false"
            unconvertible_items.append(item)
            print(f"Skipping item due to missing db_id, sql, or schema. db_id:{db_id} sql:{sql} schema:{schema}")
            continue

        # The logic to find candidates is now more powerful
        candidates = find_semantic_candidates(sql, schema)
        if candidates:
            # For simplicity, we only convert the first candidate found
            item["CONVERT"] = "true"
            item["_candidate"] = candidates[0] # Attach candidate info for the LLM
            items_to_process.append(item)
        else:
            item["CONVERT"] = "false"
            unconvertible_items.append(item)

    print(f"Found {len(items_to_process)} items that are candidates for semantic conversion.")
    
    # 4. Process convertible items with LLM
    if items_to_process:
        processed_items = process_batch_with_llm(
            items=items_to_process,
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            llm_model=os.getenv("LLM_MODEL_NAME"),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME"),
            max_workers=int(os.getenv("MAX_WORKERS"))
        )
    else:
        processed_items = []

    # 5. Combine results and save
    final_data = unconvertible_items + processed_items
    
    # Clean up the temporary '_candidate' key
    for item in final_data:
        item.pop("_candidate", None)

    print(f'''Saving {len(final_data)} processed items to {os.getenv("OUTPUT_FILE_PATH")}...''')
    with open(os.getenv("OUTPUT_FILE_PATH"), 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    
    print("--- Process Completed Successfully! ---")


if __name__ == "__main__":
    main()
