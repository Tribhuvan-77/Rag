from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredMarkdownLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from schema import Valid_Response
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from LLM import llm

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


    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100)

    chunks=splitter.split_documents(docs)

    print(type(chunks[0]))
    embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_store=Chroma(embedding_function=embedding_model,persist_directory="chromadb",collection_name="rag")
    
    vector_store.add_documents(chunks)

   #  keys=vector_store.get().keys()

    retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3,"lambda_mult":0.4})

    results=retriever.invoke(query)
   #  for i in range(len(results)):
   # #   print(f"{i}:{results[i].page_content}\n")
   # #   print(f"{i}:{results[i].metadata}\n")
    
    context=[]
    for i in results:
       context.append({"content1":i.page_content,"metadata":i.metadata})
    
    structured_llm=llm.with_structured_output(Valid_Response)
    query_prompt = """
Answer the user's question using only the provided context.

      Context:{context}
If the answer is not in the context, give a empty list for page numbers."""
    template=ChatPromptTemplate.from_messages([("system",query_prompt),("user","{query}")])
    chain=template|structured_llm

    response=chain.invoke({"query":query,"context":context})

    print(response)

    
       







    

    