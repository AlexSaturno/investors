### ====================================
### Importações e Globais
### ====================================
from pathlib import Path
import os


PASTA_RAIZ = Path(__file__).parent
PASTA_IMAGENS_DEBUG = Path(__file__).parent / "imagens_debug"
PASTA_IMAGENS = Path(__file__).parent / "files_images"
PASTA_VECTORDB = Path(__file__).parent / "vectordb"
PASTA_ARQUIVOS = Path(__file__).parent / "uploaded_files"
PASTA_RESPOSTAS = Path(__file__).parent / "respostas"
PASTA_AVALIACAO = Path(__file__).parent / "avaliacao"
if not os.path.exists(PASTA_ARQUIVOS):
    os.makedirs(PASTA_ARQUIVOS)
