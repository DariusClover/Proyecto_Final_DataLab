import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
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
        return "No se pudo generar el resumen porque el documento esta vacio."

    texto_completo = "\n\n".join([doc.page_content for doc in documents])
    llm = get_llm(temperature=0.3)

    if document_class == 'cientifico':
        template = (
            "Actua como un investigador academico experto.\n"
            "Analiza el siguiente texto cientifico y extrae estrictamente la siguiente informacion:\n"
            "1. Metodologia\n"
            "2. Hipotesis\n"
            "3. Conclusiones\n\n"
            "REGLA ESTRICTA DE FORMATO: No uses formato Markdown, ni asteriscos (* o **), ni numerales (#). "
            "Responde estrictamente en texto plano, usando solo letras, numeros, espacios y saltos de linea.\n\n"
            "Texto del documento:\n{texto}"
        )
    else:
        # Prompt Adaptativo para Documentos Generales
        template = (
            "Analiza el siguiente documento, identifica su tematica central (negocios, entretenimiento, "
            "tecnologia, cultura, etc.) y adopta el rol de un analista experto en dicha materia.\n"
            "Genera un analisis estructurado extrayendo estrictamente la siguiente informacion:\n"
            "1. Resumen Ejecutivo (o Resumen General del contenido adaptado al tema)\n"
            "2. Puntos de Accion o recomendaciones sugeridas segun la naturaleza del texto\n\n"
            "REGLA ESTRICTA DE FORMATO: No uses formato Markdown, ni asteriscos (* o **), ni numerales (#). "
            "Responde estrictamente en texto plano, utilizando unicamente letras, numeros, espacios, saltos de linea "
            "y guiones simples (-) para listar elementos si es necesario.\n\n"
            "Contenido del documento:\n{texto}"
        )

    prompt = PromptTemplate(input_variables=["texto"], template=template)
    chain = prompt | llm

    try:
        logger.info(f"Generando resumen estructurado para documento clasificado como: '{document_class}'")
        response = chain.invoke({"texto": texto_completo})
        
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
        logger.error(f"Fallo durante la generacion del resumen con Gemini. Error: {str(e)}")
        return "Error en la generacion del resumen debido a un fallo en el servicio LLM."