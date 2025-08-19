import os
import json
import sqlite3
import sqlite_vec
import csv
import re
from collections import defaultdict
import glob
from typing import List, Dict
import numpy as np
import traceback
import base64

def write_large_json(data: List[Dict], output_path: str, chunk_size: int = 500):
    """分块写入字典数组到 JSON 文件（避免嵌套数组）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')  # 开始 JSON 数组
        
        # 写入第一个元素（避免开头多余逗号）
        if len(data) > 0:
            json.dump(data[0], f, ensure_ascii=False, indent=None)
        
        # 分块写入剩余元素
        for i in range(1, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            f.write(',\n')  # 添加分隔符
            # 逐元素写入（而非整个 chunk）
            for j, item in enumerate(chunk):
                if j > 0:
                    f.write(',')
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        f.write(']')  # 结束 JSON 数组

def process_bird_dataset(base_dir="train"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, "train_tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"train_tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "train_databases")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")
        database_description = os.path.join(database_dir, "database_description")
    
        # 第一步：写入描述
        if not os.path.isdir(database_description):
                continue
        
        db_info["table_description"] = {}

        for desc in os.listdir(database_description):
            desc_path = os.path.join(database_description, desc)

            if not os.path.exists(desc_path):
                continue           

            # 表格名
            table_name = os.path.splitext(desc)[0]

            # 如果已经处理过则跳过
            if table_name in db_info["table_description"]:
                 continue
            
            try:
                # print(desc_path)
                with open(desc_path, "r", encoding="utf-8") as f:
                    db_info["table_description"][table_name] = f.read().strip()
                    # print(db_info["table_description"][table_name])
            except Exception as e:
                print(f"error in {desc}: {e}")
                pass

        db_info["table_samples"] = {}
        # 第二步 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            db_info["table_samples"][table].append(
                                dict(zip(col_names, row)))
                            # db_info["table_samples"][table][col_names] = row
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, "enhanced_train_tables.json")
    # print("Processing completed!")
    write_large_json(db_infos,output_path,2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def process_spider_dataset(base_dir="spider_data"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, "tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "database")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")
        database_description = os.path.join(database_dir, "database_description")

        db_info["table_samples"] = {}
        # 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            db_info["table_samples"][table].append(
                                dict(zip(col_names, row)))
                            # db_info["table_samples"][table][col_names] = row
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, "enhanced_train_tables_spider.json")
    # print("Processing completed!")
    write_large_json(db_infos,output_path,2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def process_spider_dataset_vector(base_dir="results"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, "new_embedding_after_add_description_tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "vector_databases_spider")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")
        database_description = os.path.join(database_dir, "database_description")

        db_info["table_samples"] = {}
        # 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            # --- 这是修改的核心部分 ---
                            processed_row = {}
                            for col_name, value in zip(col_names, row):
                                if isinstance(value, bytes):
                                    # 如果是 bytes 类型，进行 Base64 编码
                                    processed_row[col_name] = base64.b64encode(value).decode('ascii')
                                else:
                                    processed_row[col_name] = value
                            db_info["table_samples"][table].append(processed_row)
                            # --- 修改结束 ---

                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, "enhanced_new_embedding_after_add_description_tables.json")
    write_large_json(db_infos,output_path,2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")



if __name__ == "__main__":
    # base_directory = "./train"  # BIRD数据根目录
    # process_bird_dataset(base_directory)
    process_spider_dataset_vector()
