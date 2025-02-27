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
from datetime import datetime
import time
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
#            print({index}, row["Name"])
            create_date = int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())
#            create_date = convert_to_timestamp(row["CreateDate"])
            collection.add(
                ids=[f"id{index}"],
                documents=[
                    row["HostWords"]
                ],
                metadatas=[
                    {"file_name": "COA_OpenData.csv", "name": row["Name"], "type": row["Type"], "address": row["Address"], "tel": row["Tel"], "city": row["City"], "town": row["Town"], "date": create_date},
                ]
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
