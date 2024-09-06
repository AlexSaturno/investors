################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from datetime import datetime, timedelta
import time
import json
import tiktoken
import streamlit as st
from unidecode import unidecode
from time import sleep

import io
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

from utils import *

################################################################################################################################
# Ambiente
################################################################################################################################

# Parametros das APIS
# arquivo de secrets
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=st.secrets["AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"],
    model=st.secrets["AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    openai_api_type="azure",
    chunk_size=1,
)

llm = AzureChatOpenAI(
    azure_deployment=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
    model=st.secrets["AZURE_OPENAI_MODEL"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    openai_api_type="azure",
)

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


# Funcoes auxiliares
def normalize_filename(filename):
    # Mapeamento de caracteres acentuados para não acentuados
    substitutions = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Á": "A",
        "À": "A",
        "Ã": "A",
        "Â": "A",
        "Ä": "A",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "Ó": "O",
        "Ò": "O",
        "Õ": "O",
        "Ô": "O",
        "Ö": "O",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "Ç": "C",
    }

    # Substitui caracteres especiais conforme o dicionário
    normalized_filename = "".join(substitutions.get(c, c) for c in filename)

    # Remove caracteres não-ASCII
    ascii_filename = normalized_filename.encode("ASCII", "ignore").decode("ASCII")

    # Substitui espaços por underscores
    safe_filename = ascii_filename.replace(" ", "_")

    return safe_filename


def clear_respostas():
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def zera_vetorizacao():
    st.session_state["vectordb_object"] = None
    st.session_state["status_vetorizacao"] = False
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(
            f'<p class="font-stream">{result}</p>', unsafe_allow_html=True
        )


def get_stream(texto):
    for word in texto.split(" "):
        yield word + " "
        time.sleep(0.01)


# Function to initialize session state
def initialize_session_state():
    if "my_dict" not in st.session_state:
        st.session_state.my_dict = []  # Initialize as an empty list


################################################################################################################################
# UX
################################################################################################################################

# Inicio da aplicação
initialize_session_state()

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
    if "perguntas_padrao" not in st.session_state:
        st.session_state["perguntas_padrao"] = None

    if "condicoes_especiais" not in st.session_state:
        st.session_state["condicoes_especiais"] = None

    if "respostas_download_txt" not in st.session_state:
        st.session_state["respostas_download_txt"] = None

    if "respostas_download_pdf" not in st.session_state:
        st.session_state["respostas_download_pdf"] = None

    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None

    if "vectordb_object" not in st.session_state:
        st.session_state["vectordb_object"] = None

    if "status_vetorizacao" not in st.session_state:
        st.session_state["status_vetorizacao"] = False

    if "tipo_documento" not in st.session_state:
        st.session_state["tipo_documento"] = None

    if "Q&A" not in st.session_state:
        st.session_state["Q&A"] = {}

    if "Q&A_done" not in st.session_state:
        st.session_state["Q&A_done"] = False

    if "clear_respostas" not in st.session_state:
        st.session_state["clear_respostas"] = False

    if "data_processamento" not in st.session_state:
        st.session_state["data_processamento"] = None

    if "hora_processamento" not in st.session_state:
        st.session_state["hora_processamento"] = None

    if "pdf_IMG" not in st.session_state:
        st.session_state["pdf_IMG"] = None

    if "versao_prompt" not in st.session_state:
        st.session_state["versao_prompt"] = "v1"

    if "tempo_ia" not in st.session_state:
        st.session_state["tempo_ia"] = 0

    if "tempo_vetorizacao" not in st.session_state:
        st.session_state["tempo_vetorizacao"] = 0

    if "tempo_Q&A" not in st.session_state:
        st.session_state["tempo_Q&A"] = 0

    if "tempo_manual" not in st.session_state:
        st.session_state["tempo_manual"] = 0

    if "tokens_doc_embedding" not in st.session_state:
        st.session_state["tokens_doc_embedding"] = 0

    if "disable_downloads" not in st.session_state:
        st.session_state["disable_downloads"] = True

    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = True

    if "id_unico" not in st.session_state:
        st.session_state["id_unico"] = True

    username = "alesatu"

    st.subheader("Investors")
    tab1, tab2 = st.tabs(["Perguntas padrão", "Perguntas adicionais"])
    # ----------------------------------------------------------------------------------------------
    with tab1:
        with st.container():
            # Seleção do tipo de documento
            st.session_state["tipo_documento"] = st.radio(
                "Selecione o documento",
                ("Tesouro Direto", "COE", "LCA e LCI", "CDB", "Debêntures", "FAQ"),
                index=None,
                on_change=clear_respostas,
                horizontal=True,
            )
            st.write("")

            if st.session_state["tipo_documento"] is not None:
                # Upload do PDF
                with st.container(border=True):
                    pdf_file = st.file_uploader(
                        "Carregamento de arquivo",
                        type="pdf",
                        key="pdf_file",
                        on_change=zera_vetorizacao,
                    )

                if pdf_file is not None:
                    st.session_state["pdf_store"] = pdf_file.getbuffer()
                    st.session_state["file_name"] = pdf_file.name[:-4]

                    llm_call = st.button(
                        f"Processar perguntas padrão para {st.session_state.tipo_documento}"
                    )
                    ph = st.empty()

                    if llm_call:
                        data_processamento = datetime.now().strftime("%Y-%m-%d")
                        hora_processamento = (
                            datetime.now() - timedelta(hours=3)
                        ).strftime("%H:%M")
                        st.session_state["data_processamento"] = data_processamento
                        st.session_state["hora_processamento"] = hora_processamento
                        tipo = unidecode(
                            str(st.session_state["tipo_documento"]).replace(" ", "_")
                        )
                        file_name = st.session_state["file_name"]

                        id_unico = (
                            str(st.session_state["data_processamento"])
                            + "_"
                            + str(st.session_state["hora_processamento"]).replace(
                                ":", "-"
                            )
                            + "_"
                            + unidecode(str(st.session_state["file_name"]).lower())
                        )
                        st.session_state["id_unico"] = id_unico
                        pdf_store_path = str(PASTA_ARQUIVOS) + "/" + id_unico + ".pdf"

                        with open(pdf_store_path, "wb") as file:
                            file.write(st.session_state["pdf_store"])

                        st.session_state["tempo_ia"] = 0
                        start_time = time.time()

                        # Extração de texto do Pdf
                        pdf_reader = PdfReader(pdf_file)

                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=pdf_chunk,
                            chunk_overlap=pdf_overlap,
                            length_function=len,
                        )

                        chunks = text_splitter.split_text(text=text)

                        index_store_path = str(PASTA_VECTORDB) + "/" + id_unico

                        # Vetorização
                        try:
                            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                            VectorStore.save_local(index_store_path)
                            tokens_doc_embedding = num_tokens_from_string(
                                " ".join(chunks), "cl100k_base"
                            )
                            st.session_state["tokens_doc_embedding"] = (
                                tokens_doc_embedding
                            )
                            st.session_state["vectordb"] = VectorStore
                            st.session_state["status_vetorizacao"] = True
                            end_time = time.time()
                            tempo_vetorizacao = end_time - start_time
                            st.session_state["tempo_vetorizacao"] = tempo_vetorizacao
                            st.session_state["tempo_ia"] = 0

                        except Exception as e:
                            st.warning(
                                "Arquivo contém imagem e deve ser processado com OCR. Por favor subir outro arquivo."
                            )

                        # ----------------------------------------------------------------------------------------------
                        # Carrega as perguntas e faz o call para a LLM
                        with open(
                            str(PASTA_RAIZ) + "/perguntas_sidebar.json",
                            "r",
                            encoding="utf-8",
                        ) as f:
                            perguntas = json.load(f)
                        perguntas_selecionadas = list(
                            perguntas[st.session_state["tipo_documento"]].values()
                        )

                        st.session_state["selecionadas"] = perguntas_selecionadas

                        with ph.container():
                            start_time = time.time()
                            perguntas_json = st.session_state["selecionadas"]
                            total = len(perguntas_json)

                            st.session_state["Q&A"] = {}

                            with st.spinner("Processando perguntas..."):
                                for i, pergunta in enumerate(perguntas_json):
                                    tokens_query_embedding = num_tokens_from_string(
                                        pergunta, "cl100k_base"
                                    )

                                    additional_instructions_general = ".\n\nAnswers must be based on the document. The reply must be always in Portuguese from Brazil."
                                    VectorStore = st.session_state["vectordb"]
                                    docs = VectorStore.similarity_search(
                                        query=pergunta, k=k_similarity
                                    )

                                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                                    with get_openai_callback() as cb:
                                        response = chain.run(
                                            input_documents=docs,
                                            question=pergunta
                                            + additional_instructions_general,
                                        )

                                    st.session_state["Q&A"].update(
                                        {
                                            str(i + 1): {
                                                "pergunta": pergunta,
                                                "resposta_ia": response,
                                                "tokens_completion": cb.completion_tokens,
                                                "tokens_prompt": cb.prompt_tokens,
                                                "tokens_query_embedding": tokens_query_embedding,
                                                "retrieved_docs": str(docs),
                                            }
                                        }
                                    )

                                    atributos_pergunta = st.session_state["Q&A"][
                                        str(i + 1)
                                    ]

                                    pergunta_prompt = atributos_pergunta["pergunta"]
                                    resposta_llm = atributos_pergunta["resposta_ia"]

                                    if i + 1 == total:
                                        st.session_state["Q&A_done"] = True
                                        end_time = time.time()
                                        tempo_qa = end_time - start_time
                                        st.session_state["tempo_Q&A"] = tempo_qa

                    # Display do resultado e criação do dataframe de avaliação
                    if st.session_state["Q&A_done"]:
                        ph.empty()
                        sleep(0.01)
                        id_unico = st.session_state["id_unico"]

                        token_cost = {
                            "tokens_prompt": 15 / 1e6,
                            "tokens_completion": 30 / 1e6,
                            "tokens_doc_embedding": 0.001 / 1e3,
                            "tokens_query_embedding": 0.001 / 1e3,
                        }

                        with ph.container():
                            with st.container(border=True):
                                grid = st.columns([0.5, 4, 4])
                                with st.container(border=True):
                                    grid[0].markdown("**#**")
                                    grid[1].markdown("**Item**")
                                    grid[2].markdown("**Resposta IA**")

                                for i, atributos_pergunta in st.session_state[
                                    "Q&A"
                                ].items():

                                    pergunta_prompt = atributos_pergunta["pergunta"]
                                    resposta_llm = atributos_pergunta["resposta_ia"]
                                    retrieved_docs = atributos_pergunta[
                                        "retrieved_docs"
                                    ]

                                    tokens_prompt = atributos_pergunta["tokens_prompt"]
                                    tokens_completion = atributos_pergunta[
                                        "tokens_completion"
                                    ]
                                    tokens_doc_embedding = st.session_state[
                                        "tokens_doc_embedding"
                                    ]
                                    tokens_query_embedding = atributos_pergunta[
                                        "tokens_query_embedding"
                                    ]

                                    custo_prompt = (
                                        token_cost["tokens_prompt"] * tokens_prompt
                                    )
                                    custo_completion = (
                                        token_cost["tokens_completion"]
                                        * tokens_completion
                                    )
                                    custo_doc_embedding = round(
                                        token_cost["tokens_doc_embedding"]
                                        * tokens_doc_embedding,
                                        6,
                                    )
                                    custo_query_embedding = round(
                                        token_cost["tokens_query_embedding"]
                                        * tokens_query_embedding,
                                        6,
                                    )

                                    st.session_state["tempo_ia"] = (
                                        st.session_state["tempo_vetorizacao"]
                                        + st.session_state["tempo_Q&A"]
                                    )

                                    grid = st.columns([0.5, 4, 4])

                                    indice = i
                                    grid[0].markdown(indice)
                                    grid[1].markdown(pergunta_prompt)
                                    grid[2].markdown(resposta_llm)

                        if st.session_state["tempo_ia"] is not None:
                            tempo_ia = st.session_state["tempo_ia"]
                            tempo_ia = round(tempo_ia / 60, 3)
                        else:
                            tempo_ia = ""

            # ----------------------------------------------------------------------------------------------
    # Tab 2 para perguntas adicionais
    with tab2:

        def clear_text():
            st.session_state.query_add = st.session_state.widget
            st.session_state.widget = ""

        st.write("")
        if st.session_state["status_vetorizacao"]:
            st.text_input("**Digite aqui a sua pergunta**", key="widget")
            query_add = st.session_state.get("query_add", "")
            with st.form(key="myform1"):
                submit_button = st.form_submit_button(
                    label="Enviar", on_click=clear_text
                )

                if submit_button:
                    VectorStore = st.session_state["vectordb"]
                    docs = VectorStore.similarity_search(
                        query=query_add, k=k_similarity
                    )
                    st.session_state["doc_retrieval"] = docs
                    chain = load_qa_chain(llm=llm, chain_type="stuff")

                    with get_openai_callback() as cb:
                        response = chain.run(
                            input_documents=docs,
                            question=query_add
                            + ".\n\nAnswers must be based on the document. The reply must be always in Portuguese from Brazil.",
                        )
                    with st.empty():
                        st.markdown(f"**{query_add}**" + "  \n " + response)

        else:
            st.write("Documento não vetorizado!")


if __name__ == "__main__":
    main()
