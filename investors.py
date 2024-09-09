################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
import time
import tiktoken
import streamlit as st

from utils import *

from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
)

from langchain_openai.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st

################################################################################################################################
# Ambiente
################################################################################################################################
embeddings_deployment = st.secrets["AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"]
embeddings_model = st.secrets["AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME"]

azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
azure_deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
azure_model = st.secrets["AZURE_OPENAI_MODEL"]
azure_api_version = "2024-02-15-preview"  # st.secrets["AZURE_OPENAI_API_VERSION"]
azure_key = st.secrets["AZURE_OPENAI_API_KEY"]

vectorstore_address = st.secrets["AZURESEARCH_VECTORSTORE_ADDRESS"]
vectorstore_key = st.secrets["AZURESEARCH_VECTORSTORE_KEY"]


AZURESEARCH_FIELDS_ID = st.secrets["AZURESEARCH_FIELDS_ID"]
AZURESEARCH_FIELDS_CONTENT = st.secrets["AZURESEARCH_FIELDS_CONTENT"]
AZURESEARCH_FIELDS_CONTENT_VECTOR = st.secrets["AZURESEARCH_FIELDS_CONTENT_VECTOR"]

fields = [
    SimpleField(
        name=AZURESEARCH_FIELDS_ID,
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name=AZURESEARCH_FIELDS_CONTENT,
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name=AZURESEARCH_FIELDS_CONTENT_VECTOR,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
    ),
]
#############################################################################################################
# Parametros das APIS
# arquivo de secrets
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment,
    model=embeddings_model,
    azure_endpoint=azure_endpoint,
    openai_api_type="azure",
    chunk_size=1,
)

llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,
    model=azure_model,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
    api_key=azure_key,
    openai_api_type="azure",
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vectorstore_address,
    azure_search_key=vectorstore_key,
    index_name=st.secrets["AZURESEARCH_INDEX_NAME"],
    embedding_function=embeddings.embed_query,
    fields=fields,
)

#############################################################################################################
# Chat functions
PROMPT = """
You are an investors assistant. The reply must be always in Portuguese from Brazil.
Answers must be based on the vectorized database.

Context:
{context}

Current conversation:
{chat_history}

Human: {question}
"""


def cria_chain_conversa():
    memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history", output_key="answer"
    )
    retriever = vector_store.as_retriever(search_type="similarity", k=k_similarity)
    prompt = PromptTemplate.from_template(template=PROMPT)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    print("Chat chain: ", chat_chain)

    st.session_state["chain"] = chat_chain


###############################################################################################################
####################### Parametros de modelagem ###############################################################
k_similarity = 10  # lang_chain similarity search

# Tente utilizar tamanhos de chunk_sizes = [128, 256, 512, 1024, 2048]
# https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
pdf_chunk = 2048

# https://learn.microsoft.com/en-us/answers/questions/1551865/how-do-you-set-document-chunk-length-and-overlap-w
# Recomendado 10%
pdf_overlap = 205
##############################################################################################################

################################################################################################################################
# UX
################################################################################################################################

# Inicio da aplicação
st.set_page_config(
    page_title="Investors",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo css de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


################################################################################################################################
# UI
################################################################################################################################
# Inicio da aplicação
def main():
    st.subheader("Investors", divider=True)
    tab1, tab2 = st.tabs(["Assistente Investor", "Perguntas isolados"])
    # ----------------------------------------------------------------------------------------------
    with tab1:
        if not "chain" in st.session_state:
            cria_chain_conversa()

        chain = st.session_state["chain"]
        memory = chain.memory

        mensagens = memory.load_memory_variables({})["chat_history"]

        # Container para exibição no estilo Chat message
        container = st.container()
        for mensagem in mensagens:
            chat = container.chat_message(mensagem.type)
            chat.markdown(mensagem.content)

        # Espaço para o usuário incluir a mensagem e estruturação da conversa
        nova_mensagem = st.chat_input("Digite uma mensagem")
        if nova_mensagem:
            chat = container.chat_message("human")
            chat.markdown(nova_mensagem)
            chat = container.chat_message("ai")
            chat.markdown("Gerando resposta...")

            resposta = chain.invoke({"question": nova_mensagem})
            st.session_state["ultima_resposta"] = resposta
            st.rerun()

    # ----------------------------------------------------------------------------------------------
    # Tab 2 para perguntas adicionais
    with tab2:

        def clear_text():
            st.session_state.query_add = st.session_state.widget
            st.session_state.widget = ""

        st.write("")

        st.text_input("**Digite aqui a sua pergunta**", key="widget")
        query_add = st.session_state.get("query_add", "")
        with st.form(key="myform1"):
            submit_button = st.form_submit_button(label="Enviar", on_click=clear_text)

            if submit_button:
                with st.spinner("Processando..."):
                    isolated_qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=vector_store.as_retriever(
                            search_type="similarity", k=k_similarity
                        ),
                        return_source_documents=True,
                    )

                    resposta = isolated_qa.invoke(input={"query": query_add})
                    st.markdown(f"**{query_add}**" + "  \n " + resposta["result"])


if __name__ == "__main__":
    main()
