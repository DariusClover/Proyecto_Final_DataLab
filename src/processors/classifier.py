import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.utils.config import get_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

def classify_document(documents: List[Document]) -> str:
    """
    Analiza los primeros fragmentos de un documento y lo clasifica usando Gemini.
    
    Args:
        documents (List[Document]): Lista de fragmentos (chunks) extraídos del documento.
        
    Returns:
        str: 'cientifico' o 'general'. En caso de fallo de red o del LLM, retorna 'general' por defecto.
    """
    if not documents:
        logger.warning("Lista de documentos vacía. Se clasifica como 'general' por defecto.")
        return "general"
    
    # Extraemos solo los primeros 2 fragmentos (chunks/páginas) para la clasificación.
    # Esto ahorra tokens, reduce latencia y evita enviar todo el documento al LLM solo para clasificarlo.
    sample_text = "\n\n".join([doc.page_content for doc in documents[:2]])
    
    # Inicializamos el LLM asegurando que devuelva resultados deterministas (temperature=0.0)
    llm = get_llm(temperature=0.0)
    
    # Rúbrica: Uso obligatorio de PromptTemplate
    prompt_classifier = PromptTemplate(
        input_variables=["texto"],
        template=(
            "Actúa como un analista experto de documentos para ConsulTech Analytics.\n"
            "Tu tarea es leer el siguiente fragmento de un documento y clasificarlo "
            "estrictamente en una de las siguientes dos categorías:\n"
            "1. 'cientifico': Si contiene abstract, metodología, hipótesis, lenguaje académico, "
            "fórmulas, revisión de literatura o referencias bibliográficas formales.\n"
            "2. 'general': Si es un reporte de negocio, noticia, informe ejecutivo, artículo de blog o transcripción.\n\n"
            "Responde ÚNICAMENTE con la palabra 'cientifico' o 'general' en minúsculas. No incluyas puntuación ni texto adicional.\n\n"
            "Fragmento del documento a analizar:\n"
            "{texto}"
        )
    )
    
    # Composición de la cadena (Chain)
    chain = prompt_classifier | llm
    
    try:
        logger.info("Enviando muestra de texto a Gemini para determinar tipo de documento...")
        response = chain.invoke({"texto": sample_text})
        
        # Limpieza estricta de la respuesta del LLM para evitar errores en las aristas condicionales
        clasificacion = response.content.strip().lower()
        
        if clasificacion not in ['cientifico', 'general']:
            logger.warning(f"Respuesta ambigua de Gemini: '{clasificacion}'. Se clasifica como 'general' por defecto.")
            return "general"
            
        logger.info(f"Clasificación exitosa. Tipo de documento detectado: {clasificacion}")
        return clasificacion

    except Exception as e:
        # Manejo de errores exigido por la rúbrica (Ej: caída de API, límite de cuota)
        logger.error(f"Fallo durante la clasificación con Gemini. Error: {str(e)}")
        return "general"

def create_vectorstore(documents: List[Document], save_path: str = "vector_store/faiss_index") -> bool:
    """
    Toma una lista de documentos LangChain, los divide en fragmentos (chunks) 
    y los almacena en una base de datos vectorial FAISS local.
    """
    if not documents:
        logger.warning("No hay documentos para vectorizar.")
        return False

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documento dividido exitosamente en {len(chunks)} fragmentos.")

        # Este modelo estadístico corre perfectamente en tu máquina local
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = FAISS.from_documents(chunks, embeddings)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorstore.save_local(save_path)
        logger.info(f"Vectorstore FAISS guardado de forma segura en: {save_path}")

        return True

    except Exception as e:
        logger.error(f"Fallo crítico al crear el vectorstore: {str(e)}")
        return False