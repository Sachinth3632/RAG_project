from fastapi import FastAPI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

app = FastAPI()

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("vectorstore", embeddings)
llm = ChatOpenAI(model="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

@app.get("/ask")
def ask(question: str):
    return {"answer": qa.run(question)}
