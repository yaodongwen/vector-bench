# schema_utils.py

import json
import re
import hashlib
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, Set, List, Tuple

from dotenv import load_dotenv

def load_and_process_schema(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the schema file and processes it into a more accessible format.
    It now also processes semantic types from 'column_alter' and table samples.
    Returns a dictionary mapping db_id to its processed schema.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            schemas_raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {filepath}")
        return {}

    processed_schemas = {}
    for db_schema in schemas_raw:
        db_id = db_schema['db_id']
        
        table_map = {i: name for i, name in enumerate(db_schema['table_names_original'])}
        
        vector_columns: Set[str] = set()
        table_columns: Dict[str, Set[str]] = {name: set() for name in db_schema['table_names_original']}
        semantic_info: Dict[str, Dict[str, str]] = {name: {} for name in db_schema['table_names_original']}

        # Populate table_columns and vector_columns
        for col_idx, (table_idx, col_name) in enumerate(db_schema['column_names_original']):
            if table_idx >= 0 and table_idx in table_map:
                table_name = table_map[table_idx]
                table_columns[table_name].add(col_name.lower())
                if col_name.lower().endswith('_embedding'):
                    vector_columns.add(f"{table_name}.{col_name.lower()}")

        # Populate semantic_info from column_alter
        if 'column_alter' in db_schema:
            for table_name, alterations in db_schema['column_alter'].items():
                if table_name in semantic_info:
                    for alt in alterations:
                        col_name = alt.get('column_name')
                        sem_type = alt.get('semantic_type')
                        if col_name and sem_type:
                            semantic_info[table_name][col_name.lower()] = sem_type

        processed_schemas[db_id] = {
            "table_names": db_schema['table_names_original'],
            "table_columns": table_columns,
            "vector_columns": vector_columns,
            "semantic_info": semantic_info,
            "table_samples": db_schema.get('table_samples', {})
        }
        
    return processed_schemas

def get_table_aliases(sql: str) -> Dict[str, str]:
    """Extracts table aliases from a SQL query. e.g., 'movies AS T1' -> {'T1': 'movies'}"""
    aliases = {}
    pattern = re.compile(r"\b(?:FROM|JOIN)\s+([`'\"\w\.]+)\s+(?:AS\s+)?([`'\"\w]+)", re.IGNORECASE)
    matches = pattern.findall(sql)
    for table, alias in matches:
        table_name = table.split('.')[-1].strip("`'\"")
        aliases[alias.upper()] = table_name
    return aliases

def get_where_conditions(sql: str) -> List[Tuple[str, str, str]]:
    """
    Extracts simple `column = 'value'` or `column LIKE '%value%'` conditions from the WHERE clause.
    Returns a list of tuples: (full_column_name, operator, value)
    """
    conditions = []
    where_clause_match = re.search(r"WHERE\s+(.*?)(?:\sGROUP BY|\sORDER BY|\sLIMIT|\sHAVING|\sUNION|\sINTERSECT|\sEXCEPT|$)", sql, re.IGNORECASE | re.DOTALL)
    if not where_clause_match:
        return []

    where_clause = where_clause_match.group(1)
    pattern = re.compile(r"([\w\.]+)\s*(=|LIKE)\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    matches = pattern.findall(where_clause)
    
    for col, op, val in matches:
        conditions.append((col.strip(), op.strip(), val.strip()))
        
    return conditions

def find_semantic_candidates(sql: str, schema: Dict) -> List[Dict]:
    """
    Analyzes a SQL query to find opportunities for semantic conversion.
    New logic: Prioritizes converting a filter on a 'name' column to a 
    vector search on a corresponding 'description' column.
    """
    aliases = get_table_aliases(sql)
    conditions = get_where_conditions(sql)
    candidates = []
    
    semantic_info = schema.get("semantic_info", {})
    table_samples = schema.get("table_samples", {})

    for full_col, op, value in conditions:
        if '.' not in full_col:
            continue
        
        table_alias, col_name_raw = full_col.split('.', 1)
        col_name = col_name_raw.lower()
        table_name = aliases.get(table_alias.upper())

        if not table_name:
            continue

        table_semantic_info = semantic_info.get(table_name, {})
        col_semantic_type = table_semantic_info.get(col_name)

        if col_semantic_type == 'name':
            description_col = None
            for c, s_type in table_semantic_info.items():
                if s_type == 'description':
                    description_col = c
                    break
            
            if description_col:
                vector_col_name = f"{description_col}_embedding"
                full_vector_col_name = f"{table_name}.{vector_col_name}"
                
                if full_vector_col_name in schema.get("vector_columns", set()):
                    samples = table_samples.get(table_name)
                    if samples and isinstance(samples, list) and len(samples) > 0:
                        sample_description_text = samples[0].get(description_col)
                        if sample_description_text:
                            candidates.append({
                                "conversion_type": "name_to_description_vector",
                                "table_name": table_name,
                                "name_column": col_name,
                                "description_column": description_col,
                                "vector_column_name": f"{table_alias}.{vector_col_name}",
                                "sample_description_text": str(sample_description_text),
                                "original_condition": f"{full_col} {op} '{value}'"
                            })
                            return candidates
    return candidates

class LLMCache:
    # ... (LLMCache class code remains unchanged)
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any):
        self._cache[key] = value
        self._save_cache()

    def _save_cache(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)


def call_llm(client: OpenAI, model_name: str, prompt: str) -> Dict[str, str]:
    # ... (call_llm function code remains unchanged)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {}


def process_batch_with_llm(
    items: List[Dict], 
    api_key: str, 
    base_url: str, 
    llm_model: str, 
    embedding_model: str, 
    max_workers: int
) -> List[Dict]:
    """
    Processes a batch of items using the LLM with caching and parallel execution.
    """
    load_dotenv()
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = LLMCache(os.getenv("CACHE_FILE_PATH"))
    results = [None] * len(items)
    
    try:
        with open("./prompt_templates/prompt_template.txt", 'r', encoding='utf-8') as f:
            PROMPT_TEMPLATE = f.read()
    except FileNotFoundError:
        print("Error: prompt_template.txt not found in ./prompt_templates/ directory.")
        return items

    schemas = load_and_process_schema(os.getenv("SCHEMA_FILE_PATH"))

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(items):
            candidate = item.get("_candidate")
            db_id = item.get("db_id")
            if not candidate or not db_id:
                results[i] = item
                continue
            
            schema_for_db = schemas.get(db_id)
            simplified_schema_str = "Schema information is not available."

            if schema_for_db:
                table_name = candidate.get("table_name")
                if table_name:
                    # --- 核心改动：新增逻辑以包含 table_samples ---
                    table_cols = schema_for_db["table_columns"].get(table_name, set())
                    table_sem_info = schema_for_db["semantic_info"].get(table_name, {})
                    table_samples = schema_for_db["table_samples"].get(table_name, [])
                    
                    sem_cols_str_list = [f"`{col}` ({stype})" for col, stype in table_sem_info.items()]
                    
                    simplified_schema_str = f"- **Table: `{table_name}`**\n"
                    simplified_schema_str += f"  - Columns: {', '.join(sorted(list(table_cols)))}\n"
                    if sem_cols_str_list:
                        simplified_schema_str += f"  - Semantic Columns: {', '.join(sem_cols_str_list)}\n"
                    
                    # 将样例数据格式化为Markdown表格
                    if table_samples:
                        # 从第一个样本中获取表头，并排除embedding列
                        headers = [h for h in table_samples[0].keys() if not h.endswith('_embedding')]
                        
                        simplified_schema_str += "  - Sample Rows:\n"
                        simplified_schema_str += f"    | {' | '.join(headers)} |\n"
                        simplified_schema_str += f"    |{'---|' * len(headers)}\n"
                        
                        for sample_row in table_samples:
                            # 确保值为字符串并处理None
                            row_values = [str(sample_row.get(h, '')).replace('\n', ' ').replace('|', ' ') for h in headers]
                            simplified_schema_str += f"    | {' | '.join(row_values)} |\n"
                    # --- 核心改动结束 ---

            prompt = PROMPT_TEMPLATE.format(
                simplified_schema=simplified_schema_str,
                original_question=item['question'],
                original_sql=item['query'],
                original_condition=candidate['original_condition'],
                vector_column_name=candidate['vector_column_name'],
                sample_text_to_embed=candidate['sample_description_text'],
                embedding_model_name=embedding_model
            )
            
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cached_result = cache.get(prompt_hash)
            
            if cached_result:
                processed_item = {**item}
                processed_item["query"] = cached_result.get("new_sql", item["query"])
                processed_item["question"] = cached_result.get("new_question", item["question"])
                results[i] = processed_item
            else:
                future = executor.submit(call_llm, client, llm_model, prompt)
                futures_map[future] = (i, prompt_hash)

        if not futures_map:
             return [res for res in results if res is not None]

        for future in tqdm(as_completed(futures_map), total=len(futures_map), desc="Generating Semantic Pairs"):
            index, prompt_hash = futures_map[future]
            original_item = items[index]
            try:
                llm_result = future.result()
                if llm_result and "new_sql" in llm_result:
                    cache.set(prompt_hash, llm_result)
                    processed_item = {**original_item}
                    processed_item["query"] = llm_result.get("new_sql")
                    processed_item["question"] = llm_result.get("new_question", original_item["question"])
                    results[index] = processed_item
                else:
                    results[index] = original_item 
            except Exception as e:
                print(f"Future resulted in an exception: {e}")
                results[index] = original_item

    return [res for res in results if res is not None]
