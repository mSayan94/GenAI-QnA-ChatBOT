import os
from pinecone import Pinecone, PodSpec
import time

# Configuring client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = 'langchainvector'
spec = PodSpec(environment="gcp-starter")

def create_vector_index(index_name = index):
    # Check for and delete index if already exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create a new index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
        spec=spec
    )

    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    return index_name