################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
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
azure_api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
azure_key = st.secrets["AZURE_OPENAI_API_KEY"]

vectorstore_address = st.secrets["INVESTORS_AZURESEARCH_VECTORSTORE_ADDRESS"]
vectorstore_key = st.secrets["INVESTORS_AZURESEARCH_VECTORSTORE_KEY"]

INVESTORS_AZURESEARCH_FIELDS_CHUNK_ID = "chunk_id"
INVESTORS_AZURESEARCH_FIELDS_PARENT_ID = "parent_id"
INVESTORS_AZURESEARCH_FIELDS_CHUNK = "chunk"
INVESTORS_AZURESEARCH_FIELDS_TITLE = "title"
INVESTORS_AZURESEARCH_FIELDS_TEXT_VECTOR = "text_vector"

AZURESEARCH_FIELDS_CHUNK_ID = st.secrets["INVESTORS_AZURESEARCH_FIELDS_CHUNK_ID"]
AZURESEARCH_FIELDS_PARENT_ID = st.secrets["INVESTORS_AZURESEARCH_FIELDS_PARENT_ID"]
AZURESEARCH_FIELDS_CHUNK = st.secrets["INVESTORS_AZURESEARCH_FIELDS_CHUNK"]
AZURESEARCH_FIELDS_TITLE = st.secrets["INVESTORS_AZURESEARCH_FIELDS_TITLE"]
AZURESEARCH_FIELDS_TEXT_VECTOR = st.secrets["INVESTORS_AZURESEARCH_FIELDS_TEXT_VECTOR"]

fields = [
    SimpleField(
        name=AZURESEARCH_FIELDS_CHUNK_ID,
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SimpleField(name=AZURESEARCH_FIELDS_PARENT_ID, type=SearchFieldDataType.String),
    SimpleField(name=AZURESEARCH_FIELDS_CHUNK, type=SearchFieldDataType.String),
    SimpleField(name=AZURESEARCH_FIELDS_TITLE, type=SearchFieldDataType.String),
    SearchField(
        name=AZURESEARCH_FIELDS_TEXT_VECTOR,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
    ),
]
#############################################################################################################
# Parâmetros das APIs
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
    index_name=st.secrets["INVESTORS_AZURESEARCH_INDEX_NAME"],
    embedding_function=embeddings.embed_query,
    fields=fields,
)

#############################################################################################################
# Funções do Chat
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
    # Defina chaves únicas para a memória e a cadeia desta página
    memory_key = "memory_investors"
    chain_key = "chain_investors"

    # Verifique se a memória já existe no session_state
    if memory_key not in st.session_state:
        st.session_state[memory_key] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
        )
    memory = st.session_state[memory_key]

    retriever = vector_store.as_retriever(search_type="similarity", k=k_similarity)
    prompt = PromptTemplate.from_template(template=PROMPT)

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    st.session_state[chain_key] = chat_chain


###############################################################################################################
####################### Parâmetros de modelagem ###############################################################
k_similarity = 10  # lang_chain similarity search

# Tamanhos de chunk_size recomendados
pdf_chunk = 2048

# Sobreposição recomendada de 10%
pdf_overlap = 205
##############################################################################################################

################################################################################################################################
# UX
################################################################################################################################

# Início da aplicação
st.set_page_config(
    page_title="Investors",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo CSS de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


################################################################################################################################
# UI
################################################################################################################################
def main():
    chain_key = "chain_investors"
    memory_key = "memory_investors"

    if chain_key not in st.session_state:
        cria_chain_conversa()

    chain = st.session_state[chain_key]
    memory = st.session_state[memory_key]

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


if __name__ == "__main__":
    main()
