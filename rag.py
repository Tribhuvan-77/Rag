from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredMarkdownLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from schema import Valid_Response
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from LLM import llm
from uuid import uuid4

def Load_File(filepath:str,filetype:str,query:str):
    if filetype==".pdf":
       loader=PyPDFLoader(file_path=filepath)
    elif filetype==".txt":
       
       loader=TextLoader(file_path=filepath)
    elif filetype==".docx":
       loader=UnstructuredWordDocumentLoader(file_path=filepath)
    else:
       loader=UnstructuredMarkdownLoader(file_path=filepath)
    
    docs=loader.load()


    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=0)

    chunks=splitter.split_documents(docs)

    print(type(chunks[0]))
    embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    collection=str(uuid4())
    vector_store=Chroma(embedding_function=embedding_model,persist_directory="chromadb",collection_name=collection)

    vector_store.add_documents(chunks)
    print(vector_store._collection.count())

   #  keys=vector_store.get().keys()

    retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":5,"lambda_mult":0.5})

    results=retriever.invoke(query)
    for i in range(len(results)):
     print(f"{i}:{results[i].page_content}\n")
     print(f"{i}:{results[i].metadata}\n")
    
    context=[]
    for i in results:
       context.append({"content1":i.page_content,"metadata":i.metadata})
    
    structured_llm=llm.with_structured_output(Valid_Response)
    query_prompt = """
You are a helpful AI assistant.

Understand the provided context carefully and answer the user's question using **only** the information in the context.

- Do not use external knowledge or make assumptions.
- If the answer is fully supported by the context, provide a clear and concise answer.
- If the context does not contain enough information, state that the answer is not available in the provided context.
- If the answer is not found, return an empty list for page numbers.
- Include only the page numbers that directly support your answer. Do not guess or invent page numbers.

Context:
{context}"""
    template=ChatPromptTemplate.from_messages([("system",query_prompt),("user","{query}")])
    chain=template|structured_llm

    response=chain.invoke({"query":query,"context":context})

    print(response)

    
       







    

    