import langchain
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import chroma
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import nltk
os.environ["OPENAI_API_KEY"]="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"


def Chat_bot(document_path,question):
        loader=UnstructuredFileLoader(document_path)
        documents=loader.load()
        text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        texts=text_splitter.split_documents(documents)
        embeddings=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        doc_search=chroma.from_documents(texts,embeddings)
        chain=VectorDBQA.from_chain_type(m=OpenAI(),chain_type="stuff",vectorstore=doc_search)
        print(chain.run(question))
set_document_path=("Automate_the_Boring_Stuff_with_Python_Practical_Programming_for.pdf")
while True:
    print("you can ask any questions from the given document \n if you want to exit tupe exit".upper())
    question=input("\n Ask me Question: ".upper())
    if question=="exit":
          print("thank you :) bye !!!")
          break
    else:
        Chat_bot(set_document_path,question)