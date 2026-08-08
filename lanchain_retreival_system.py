from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.chains import RetrievalQA

# 1. Load a document about AI
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents = loader.load()

# 2. Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
# Apply both splitters to the PDF document
chunks = text_splitter.split_documents(documents)

# 3. Set up the embedding model. (Use an embedding model to create vector representations.)
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
from langchain_ibm import WatsonxEmbeddings

embedding_model = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)


# 4. Create a vector store
vector_store = Chroma(
    collection_name="split_parents", embedding_function=embedding_model
)
vector_store.add_documents(chunks)
# 5. Create a retriever
retriever = vector_store.as_retriever()
# 6. Define a function to search for relevant information
def search_documents(query, top_k=3):
    """Search for documents relevant to a query"""
    # Use the retriever to get relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Limit to top_k if specified
    return docs[:top_k]

# 7. Test with a few queries
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = search_documents(query)
    print(results)
