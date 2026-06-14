import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

def generate_summary(documents: List[Document], document_class: str) -> str:
    """
    Genera un resumen adaptado al tipo de documento utilizando Gemini.
    
    Args:
        documents (List[Document]): Los fragmentos del documento cargado.
        document_class (str): 'cientifico' o 'general'.
        
    Returns:
        str: El texto del resumen generado por el LLM.
    """
    if not documents:
        logger.warning("No hay documentos para resumir.")
        return "No se pudo generar el resumen porque el documento está vacío."

    # Unimos el texto de los fragmentos. Para documentos muy grandes, en producción
    # se usaría map-reduce, pero para esta prueba uniremos el texto directamente.
    texto_completo = "\n\n".join([doc.page_content for doc in documents])
    
    # Inicializamos el LLM. Usamos una temperatura ligera (0.3) para permitir 
    # naturalidad en la redacción del resumen sin perder precisión.
    llm = get_llm(temperature=0.3)

    # Definición de Prompts según rúbrica (Fase 3)
    if document_class == 'cientifico':
        template = (
            "Actúa como un investigador académico experto.\n"
            "Analiza el siguiente texto científico y extrae estrictamente la siguiente información "
            "estructurada con claridad:\n"
            "1. Metodología\n"
            "2. Hipótesis\n"
            "3. Conclusiones\n\n"
            "Texto del documento:\n{texto}"
        )
    else:
        template = (
            "Actúa como un consultor senior de negocios para ConsulTech Analytics.\n"
            "Analiza el siguiente reporte corporativo y extrae estrictamente la siguiente información "
            "estructurada con claridad:\n"
            "1. Resumen Ejecutivo\n"
            "2. Puntos de Acción (Action Points) recomendados\n\n"
            "Documento corporativo:\n{texto}"
        )

    prompt = PromptTemplate(input_variables=["texto"], template=template)
    chain = prompt | llm

    try:
        logger.info(f"Generando resumen estructurado para documento clasificado como: '{document_class}'")
        response = chain.invoke({"texto": texto_completo})
        
        # Corrección de tipo: Validar si la respuesta viene empaquetada como lista
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            content = "".join(text_parts)

        logger.info("Resumen generado exitosamente.")
        return content.strip()

    except Exception as e:
        logger.error(f"Fallo durante la generación del resumen con Gemini. Error: {str(e)}")
        return "Error en la generación del resumen debido a un fallo en el servicio LLM."