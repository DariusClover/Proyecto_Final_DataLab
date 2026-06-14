import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

# Configuración básica de logging para registrar errores en consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List[Document]:
    """
    Función que actúa como los 'sentidos' del agente.
    Lee un archivo basado en su extensión y retorna una lista de documentos de LangChain.
    
    Args:
        file_path (str): La ruta absoluta o relativa del archivo a leer.
        
    Returns:
        List[Document]: Lista de objetos Document con el contenido y metadatos.
                        Retorna una lista vacía si hay un error.
    """
    
    #  Validación de existencia del archivo
    if not os.path.exists(file_path):
        logger.error(f"Error de lectura: El archivo no existe en la ruta {file_path}")
        return []

    #  Extracción de la extensión del archivo en minúsculas
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    logger.info(f"Intentando procesar archivo de tipo: {file_extension}")
    
    documents = []
    
    # Lógica de enrutamiento y manejo estricto de errores (Requisito de Rúbrica)
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
        elif file_extension == '.txt':
            # Se especifica utf-8 para evitar errores con caracteres especiales en español
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
        elif file_extension == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
        else:
            logger.warning(f"Formato no soportado: {file_extension}. Formatos válidos: PDF, DOCX, TXT, CSV.")
            return []
            
        logger.info(f"Éxito: Se extrajeron {len(documents)} fragmentos/páginas del documento.")
        return documents

    except Exception as e:
        # Captura cualquier error de lectura (ej. archivo corrupto) para que el agente no "muera"
        logger.error(f"Fallo crítico al procesar el archivo {file_path}. Detalle del error: {str(e)}")
        return []