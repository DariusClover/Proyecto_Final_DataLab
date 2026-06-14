import os
import logging
from src.agent.state import AgentState
from src.processors.document_loader import load_document
from src.processors.classifier import classify_document
from src.generators.summarizer import generate_summary
from src.generators.pdf_generator import generate_pdf_report

logger = logging.getLogger(__name__)

def analyze_document_batch_node(state: AgentState) -> AgentState:
    """
    Recorre el lote de archivos en el estado, aplicando de forma aislada
    el pipeline de carga, clasificación y generación de resúmenes.
    """
    logger.info("--- NODO INICIADO: Analizando Lote Multi-Documento ---")
    file_paths = state.get("file_paths", [])
    analyses_accumulated = []

    if not file_paths:
        logger.error("No se encontraron rutas de archivos para procesar.")
        state["status"] = "error_sin_archivos"
        return state

    for path in file_paths:
        file_name = os.path.basename(path)
        logger.info(f"Iniciando procesamiento individual: {file_name}")
        
        try:
            # 1. Ingesta multi-formato
            docs = load_document(path)
            if not docs:
                logger.warning(f"Archivo sin contenido extraíble: {file_name}")
                continue
            
            # 2. Clasificación automática (Tu función con Gemini)
            doc_class = classify_document(docs)
            
            # 3. Resumen Estructurado (Tu función de la Fase 3)
            summary = generate_summary(docs, doc_class)
            
            # Almacenamiento de metadatos y resultados por documento
            analyses_accumulated.append({
                "file_name": file_name,
                "document_class": doc_class,
                "summary": summary,
                "status": "exitoso"
            })
            
        except Exception as e:
            logger.error(f"Fallo al procesar el documento {file_name}. Error: {str(e)}")
            analyses_accumulated.append({
                "file_name": file_name,
                "document_class": "error",
                "summary": f"No se pudo analizar el archivo debido a un error técnico: {str(e)}",
                "status": "fallido"
            })

    # Seteamos las variables consolidadas en el estado maestro
    state["analyses"] = analyses_accumulated
    state["status"] = "documentos_analizados"
    return state

def generate_pdf_report_node(state: AgentState) -> AgentState:
    """
    Nodo final que procesa la colección de análisis y construye el reporte de cierre.
    """
    logger.info("--- NODO INICIADO: Generando PDF Consolidado ---")
    return generate_pdf_report(state)