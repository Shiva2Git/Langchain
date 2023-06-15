from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
import nltk
import magic

os.environ["OPENAI_API_KEY"]="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"

loader=DirectoryLoader('Automate_the_Boring_Stuff_with_Python_Practical_Programming_for.pdf',glob='**/*.pdf')
docs=loader.load()

char_text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)

doc_text=char_text_splitter.split_documents(docs)

OpenAI_embeddings=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

vStore=chroma.from_documents(doc_text,OpenAI_embeddings)

model=VectorDBQA.from_chain_type(m=OpenAI(),chain_type="stuff",vectorstore=vStore)


q="what is python ?"
print(model.run(q))



