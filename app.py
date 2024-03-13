# Importing Libraries
import os
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from vectordb import create_vector_index
load_dotenv()

# Reading the Document
def read_docs(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
docs = read_docs("docs/")
len(docs)

# Split the docs into Chunks
def create_chunks(docs, chunk_size = 800, chunk_overlap = 50):
    textsplitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    doc = textsplitter.split_documents(docs)
    return doc
chunks=create_chunks(docs=docs)
len(chunks)

# Create Embeddings

# Initialize Azure Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"), 
    model=os.getenv("OPENAI_EMBEDDING_MODEL"), 
    api_key=os.getenv("OPENAI_API_KEY")
)

# Converting text to Vectors
vectors = embeddings.embed_query("How are you")
len(vectors)

# Vector DB in Pinecone

# Configuring Pinecone client
index_name = create_vector_index()
pinecone_index = PineconeVectorStore.from_documents(docs,embeddings,index_name=index_name)

# Cosine Similiarity: retrieve results from Vector DB
def retrieve_query(query, k=2):
    matching_results = pinecone_index.similarity_search(query, k=k)
    return matching_results

# Loading OpenAI and QA Chains



# Get QA Chain
def get_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I regret to inform you that I currently lack the necessary data to address your inquiry. Perhaps there's another way I can be of assistance?", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Initializing LLM Model
    azure_chat_llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"), 
        model=os.getenv("OPENAI_DEPLOYMENT_MODEL"), 
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5
    )

    custom_prompt = PromptTemplate(template = prompt_template, input_variables =["context","question"])
    qa_chain = load_qa_chain(azure_chat_llm, chain_type="stuff",prompt = custom_prompt)
    return qa_chain

# Search answers in VectorDB
def retrieve_answers(query):
    docs = retrieve_query(query)
    chain = get_qa_chain()
    # response = chain.run(input_documents = doc_search, question = query)
    response = chain.invoke({"input_documents":docs, "question":query}, return_only_outputs = True)
    return response
