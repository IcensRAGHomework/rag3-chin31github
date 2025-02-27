import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

#must comment rich when upload to homework
#from rich import print as pprint

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

import csv
import datetime
import time
import pandas as pd

def generate_hw08():
    # 連接地端的database
    chroma_client = chromadb.PersistentClient(path=dbpath)
    # 建立embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    # 建立collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        # 讀取CSV檔案
        csv_file_name = "COA_OpenData.csv"
        df = pd.read_csv(csv_file_name)
        print("columns: "+df.columns)

        for idx, row in df.iterrows():
            metadata = {
                "file_name": csv_file_name,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())  # 轉timeStamp
            }
            print(str(idx)+str(metadata))
            print("\n")
            # 將資料寫入 ChromaDB
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
    return collection
    pass
    
def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    def convert_to_timestamp(date_str):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return int(time.mktime(dt.timetuple()))
        except ValueError:
            return None
    with open('COA_OpenData.csv', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            print(str(index), row["Name"])
#            create_date = int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())
#            create_date = convert_to_timestamp(row["CreateDate"])
            create_date = int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())
            metas = {"file_name": "COA_OpenData.csv", "name": row["Name"], "type": row["Type"], "address": row["Address"], "tel": row["Tel"], "city": row["City"], "town": row["Town"], "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())}
            collection.add(
                ids=[str(index)],
                metadatas=[metas],
                documents=[row["HostWords"]]
            )

    return collection
    pass
    
def generate_hw02(question, city, store_type, start_date, end_date):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    results = collection.query(
        query_texts=["我想要找有關茶餐點的店家"],
        n_results=10,
        where={"city": "宜蘭縣"}
    )
    return results
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    return collection
    pass

generate_hw01()
#pprint(generate_hw02("question", "city", "store_type", "start_date", "end_date"))
