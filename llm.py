
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.documents  import Document
from langchain_core.prompts import PromptTemplate
from typing import Iterable
def format_docs(docs: Iterable[Document]):
    return '\n\n'.join([doc.page_content for doc in docs])


def llm_chat(llm, retriever,  question:str):
    
    # Reply_to: {request.reply_to}
    # context: {request.tone}
    # length: {request.len}
    prompt = PromptTemplate.from_template("""
    Given information following: 
    --------------------
    Context: {context}
    --------------------
    Question: {question}
    --------------------
    Answer: 
    """)
 

    rag_llm = (
        {"context": retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )
    response = rag_llm.invoke(question)
    return response