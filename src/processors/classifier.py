import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.utils.config import get_llm

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

# -----------------------------------------------------------------------------------------
# A partir de aquí vas tú we.
# Él deberá agregar debajo de esta línea su función para vectorizar los documentos con FAISS.
# -----------------------------------------------------------------------------------------