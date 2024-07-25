import streamlit as st
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LOAD THE VECTOR DATABASE AND PREPARE RETRIEVAL
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
retriever = vectorstore.as_retriever()

# PREPARE PROMPT
system_prompt = (
    "You are a friendly clinician at the National Institutes of Health (NIH)."
    "Your task is to answer clients questions as truthfully as possible."
    "Use the provided retrieved information from the NIH website to "
    "help answer the questions. Your performance is critical to the "
    "health of the clients and the success of your career."
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# PAGE
st.title("🤖⚕️ NIH RAG Demo")

st.info(
    "This is a chatbot that uses the NIH ODS Fact Sheets (https://ods.od.nih.gov/api/) to generate medically grounded answers."
)

query = st.chat_input(placeholder="Your search query...")

if query:
    with st.chat_message("user"):
        st.write(query)

    result = rag_chain.invoke({"input": query})
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.write("**Sources:**")
        for doc in result["context"]:
            st.write(f'- {doc.metadata["source"].replace(" ", "%20")}')
