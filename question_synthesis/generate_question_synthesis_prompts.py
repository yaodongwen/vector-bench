import json
import os
import random
import sqlite3
import sqlite_vec
import sqlite_lembed
import numpy as np
import re
from tqdm import tqdm

# 增强的风格描述，添加向量查询示例
style2desc = {
"Formal": '''**Formal Style**
   - Uses standard grammar and vocabulary.
   - Example: Find all students older than 18 years and return their home addresses.
   - Vector Example: Find the three articles most closely related to Stable Diffusion and return them.''',

"Colloquial": '''**Colloquial Style**
   - Employs informal vocabulary and expressions.
   - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.
   - Vector Example: Hey there! Can you grab me the top 3 articles that are most closely related to Stable Diffusion?''',

"Imperative": '''**Imperative Style**
   - Uses command or directive sentences.
   - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!
   - Vector Example: Please find the three articles most closely related to Stable Diffusion and return their name.''',

"Interrogative": '''**Interrogative Style**
   - Uses question forms.
   - Example: Could you tell me which students are older than 18 and what their home addresses are?
   - Vector Example: Could you show me the 3 articles that most have to do with Stable Diffusion?''',

"Descriptive": '''**Descriptive Style**
   - Uses detailed descriptions with contextual information.
   - Example: I want to know the names and home addresses of all students older than 18.
   - Vector Example: I need to find articles that most closely related to Stable Diffusion, returning the top 3 matches sorted by cosine similarity.''',

"Concise": '''**Concise Style**
   - Use short sentences.
   - Example: Students older than 18, return their names and addresses.
   - Vector Example: Top 3 related articles to Stable Diffusion.''',

"Vague": '''**Vague Style**
   - Includes ambiguous vocabulary requiring inference.
   - Example: What are the names and addresses of those older students? (External Knowledge: 'older students' refers to age >= 18.)
   - Vector Example: Find a few articles have to do with Stable Diffusion. (External Knowledge: 'a few' refers to vector similarity search with k=3 limit)''',

"Metaphorical": '''**Metaphorical Style**
   - Uses metaphors or metaphorical expressions.
   - Example: Find the names and addresses of those who have reached adulthood. (External Knowledge: 'reached adulthood' refers to age >= 18.)
   - Vector Example: Find a few articles have to do with SD in ai. (External Knowledge: 'SD in ai' refers to Stable Diffusion)''',

"Multi-turn Dialogue": '''**Multi-turn Dialogue Style**
    - This involves a dialogue to clarify the user's query needs.
    - Example: [{"User": "I want to query some student information."}, {"Assistant": "Which students' information would you like to query?"}, {"User": "Students older than 18."}, {"Assistant": "What other information would you like to know about them?"}, {"User": "Names and addresses."}, {"Assistant": "Is there anything else you need?"}, {"User": "No."}, {"Assistant": "OK, I will help you translate your request into an SQL query."}]
    - Vector Example: 
      User: "I'm looking for some articles."
      Assistant: "How many articles would you like to find and What field of paper are you looking for?"
      User: "About 3, and they are related to Stable Diffusion."
      Assistant: "I'll search for 3 articles that most closely related to Stable Diffusion."'''
}

# 增强的步骤说明，添加向量查询处理
steps_wo_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.'''

steps_w_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.
3. **External Knowledge:** For Vague or Metaphorical styles, include external knowledge to enhance clarity, especially for vector operations.'''

steps_multi_round = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Dialogue:** Create a conversation between the User and the Assistant based on the SQL query and its explanation, ensuring vector operations are properly discussed.'''

# 增强的指导方针，添加向量查询专用说明
guidelines_wo_ek = '''1. Clearly describe the columns being selected by the SQL query. For example:
   - "SELECT * ... FROM ..." means "Find all ...";
   - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
   - "SELECT rowid, vec FROM vec_table WHERE vec MATCH lembed(_,"xxx") ORDER BY distance LIMIT 2;" means "Return two of the rowid and vec that most related to xxx from vec_table, ordered by similarity distance".
   - "SELECT rowid, vec FROM vec_table WHERE vec MATCH lembed(_,"xxx") AND k = 2;" means "Return two of the rowid and vec that most related to xxx from vec_table, ordered by similarity distance".
   - For vector searches: Always mention the LIMIT value or K value when explaining MATCH operations.

2. Ensure the natural language question accurately captures:
   - All conditions including vector similarity searches
   - ORDER BY clauses (especially for distance/similarity)
   - LIMIT and K clauses
   - Any window functions or complex joins'''

guidelines_w_ek = '''1. Clearly describe the columns being selected by the SQL query (same as above).
2. Ensure the natural language question captures all query semantics (same as above).
3. For vector searches, include these common external knowledge points:
   - "MATCH" operator performs approximate nearest neighbor (ANN) search;
   - "k=N" specifies the number of similar items to return;
   - Vectors are compared using Euclidean distance (L2 norm) by default;
   - Similarity increases as distance decreases;
   - Include any domain-specific knowledge about the vector meaning.'''

guidelines_multi_round = '''1. Clearly describe the columns being selected by the SQL query (same as above).
2. Ensure the dialogue naturally covers:
   - The purpose of the vector search;
   - How many similar items are needed (LIMIT);
   - What the target vector represents;
   - Any additional filtering or sorting requirements.'''

# NEW: Specific guidelines for generating natural questions from vector queries. This is the core of the new logic.
vector_question_guidelines = '''

**Crucial Rule for Vector Queries**: When generating a question for a SQL query that contains a `lembed()` function, you must follow this rule to ensure the question is natural and useful for training data.

1.  **Abstract and Rephrase the Intent**: Analyze the text inside the `lembed()` function (e.g., `lembed('model', 'some descriptive text')`). This text is a highly specific description. Your task is to generate a question that reflects a user's *intent* behind this search, not the specific text itself.
2.  **Strictly Prohibit Copying**: You are **strictly forbidden** from using the exact string literal from the `lembed()` function in the natural language question. The question must be a paraphrase, a summary, or a more general expression of the concept.

---
**Example of Correct vs. Incorrect Behavior:**

* **Input VecSQL**: `SELECT episode_id FROM Episode WHERE summary_embedding MATCH lembed('all-MiniLM-L6-v2', "An intriguing episode with unexpected plot twists and character development") LIMIT 5;`

* **BAD Question (Incorrect Generation)**: `"I need to find the IDs of the top 5 episodes that are most closely related to the description \"An intriguing episode with unexpected plot twists and character development\"..."`
    * **Reasoning**: This is bad because it's unnatural and directly copies the search text.
    
* **GOOD Question (Correct Generation)**: `"What are some episodes with surprising plot twists and good character arcs?"` or `"Recommend 5 of the most unexpected episodes."`
    * **Reasoning**: This is good because it captures the *semantic essence* of the search text in a natural, human-like way.
---'''


# 增强的输出格式，添加向量查询字段
output_format_wo_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations if present)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question capturing all query elements)
[QUESTION-END]'''

output_format_w_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question)
[QUESTION-END]

[EXTERNAL-KNOWLEDGE-START]
(Relevant knowledge about vector operations and domain context)
[EXTERNAL-KNOWLEDGE-END]'''

output_format_multi_round = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations)
[EXPLANATION-END]

[QUESTION-START]
(Multi-turn dialogue covering vector search details)
[QUESTION-END]'''

# 增强的指令，添加向量查询检查
instruction_wo_ek = '''Based on the above information:
1. Analyze the SQL query, paying special attention to any vector operations
2. Generate a clear explanation covering all query elements
3. Formulate a precise natural language question
4. Verify all vector operations (MATCH, LIMIT, ORDER BY distance) or (MATCH, And k = ?) are properly represented'''

instruction_w_ek = '''Based on the above information:
1. Analyze the SQL query, especially vector operations
2. Generate explanation covering all elements
3. Formulate precise question
4. Add relevant external knowledge about vector operations
5. Verify all vector elements are properly represented'''

instruction_multi_round = '''Based on the above information:
1. Analyze the SQL query, especially vector operations
2. Generate explanation covering all elements
3. Create natural dialogue that explores vector search parameters
4. Ensure LIMIT, target vector and distance sorting are discussed'''


def contains_virtual_table(statements):
    """
    检测 statements 中是否包含虚拟表
    
    参数:
        statements: 包含 SQL 语句的列表或字符串
    
    返回:
        bool: 如果检测到虚拟表返回 True，否则返回 False
    """
    if isinstance(statements, str):
        statements = [statements]
    
    # 定义虚拟表关键词模式（不区分大小写）
    patterns = [
        r'\bvirtual\b',  # 匹配 "virtual" 单词
        r'\bvec0\b',     # 匹配 "vec0" 单词
        r'_embedding\b',  # 匹配以 "_embedding" 结尾的单词
        r'\bfloat\[',
        r'\]\b'       #匹配float类型的向量
    ]
    
    for stmt in statements:
        if not stmt:
            continue
            
        # 检查每个关键词模式
        for pattern in patterns:
            if re.search(pattern, stmt, re.IGNORECASE):
                return True
                
    return False

def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 
    sqlite_lembed.load(conn)

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_names = []
    create_statements = []
    for table in tables:
        table_name, create_statement = table
        table_names.append(table_name)
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return table_names, create_statements

def extract_column_descriptions(create_statements):
    column_name2column_desc = dict()
    # 匹配常规列和向量列
    pattern = r'"(\w+)"\s+(\w+)\s*/\*\s*(.*?)\s*\*/'
    
    for create_statement in create_statements:
        matches = re.findall(pattern, create_statement)
        for column_name, col_type, description in matches:
            column_name = column_name.lower()
            # 特殊处理向量列
            if col_type.upper() in ('VECTOR', 'FLOAT[]', 'FLOAT[%d]'):
                desc = f"Vector column: {description}" if description else "Floating-point vector column"
            else:
                desc = description if description else f"{col_type} column"
            column_name2column_desc[column_name] = desc

    return column_name2column_desc

def detect_vector_columns(sql):
    """检测SQL中是否包含向量查询"""
    # 匹配两种格式：
    # 1. MATCH '[0.1, 0.2, ...]'
    # 2. MATCH lembed('model_name', 'text')
    pattern = r'MATCH\s+(?:lembed\([^)]+\)|\'\[[^\]]+\]\')'
    return bool(re.search(pattern, sql, re.IGNORECASE))

def enhance_vector_info(sql, column_info):
    """增强向量列的描述信息"""
    if detect_vector_columns(sql):
        for col in list(column_info.keys()):
            if '_embedding' in col:
                column_info[col] = f"Vector column for similarity search: {column_info.get(col, '')}"
    return column_info

if __name__ == "__main__":
    random.seed(42)
    db_path = "../brid_vectorization/results/vector_databases_brid"
    sql_infos = json.load(open("../sql_synthesis/results/synthetic_sqls.json"))
    question_synthesis_template = open("./prompt_templates/question_synthesis_prompt.txt").read()
    styles = list(style2desc.keys())

    # 创建输出目录
    os.makedirs("./prompts", exist_ok=True)

    db_ids = list(set([sql["db_id"] for sql in sql_infos]))
    db_id2column_info = dict()
    db_id_vec_flag = dict()
    
    # 获取数据库模式信息
    for db_id in tqdm(db_ids, desc="Processing databases"):
        table_names, create_statements = obtain_db_schema(os.path.join(db_path, db_id, db_id + ".sqlite"))
        db_id_vec_flag[db_id] = contains_virtual_table(create_statements)
        db_id2column_info[db_id] = extract_column_descriptions(create_statements)
    
    prompts = []
    for sql_info in tqdm(sql_infos, desc="Generating prompts"):
        style_name = random.choice(styles)
        column_info = db_id2column_info[sql_info["db_id"]]
        
        # 增强向量列信息
        column_info = enhance_vector_info(sql_info["sql"], column_info)
        
        # 只保留SQL中实际使用的列
        used_columns = {}
        sql_lower = sql_info["sql"].lower()
        for col_name, col_desc in column_info.items():
            if col_name.lower() in sql_lower:
                used_columns[col_name] = col_desc

        # 选择适当的模板变体
        if style_name in ["Vague", "Metaphorical"]:
            steps = steps_w_ek
            guidelines = guidelines_w_ek
            instruction = instruction_w_ek
            output_format = output_format_w_ek
        elif style_name == "Multi-turn Dialogue":
            steps = steps_multi_round
            guidelines = guidelines_multi_round
            instruction = instruction_multi_round
            output_format = output_format_multi_round
        else:
            steps = steps_wo_ek
            guidelines = guidelines_wo_ek
            instruction = instruction_wo_ek
            output_format = output_format_wo_ek
        
        # MODIFIED: Conditionally append the new vector guidelines
        is_vector_query = detect_vector_columns(sql_info["sql"])
        if is_vector_query:
            guidelines += vector_question_guidelines
            
        # MODIFIED: Corrected the logic for defining prompt_with_vector for robustness
        prompt_with_vector = ""
        if db_id_vec_flag.get(sql_info["db_id"], False):
            prompt_with_vector = "You have to use KNN queries, if extension includes sqlite-vec."
        
        # 生成提示词
        prompt = question_synthesis_template.format(
            using_knn=prompt_with_vector,
            style_desc=style2desc[style_name].strip(),
            engine="SQLite",
            extension="sqlite-vec and sqlite-lembed",
            column_info=json.dumps(used_columns, indent=2, ensure_ascii=False).strip(),
            sql=sql_info["sql"].strip(),
            steps=steps.strip(),
            guidelines=guidelines.strip(),
            output_format=output_format.strip(),
            instruction=instruction.strip()
        )
        
        sql_info["style"] = style_name
        sql_info["prompt"] = prompt
        # MODIFIED: Use the flag we already calculated
        sql_info["contains_vector"] = is_vector_query
    
    # 保存结果
    with open("./prompts/question_synthesis_prompts.json", "w", encoding="utf-8") as f:
        json.dump(sql_infos, f, indent=2, ensure_ascii=False)
