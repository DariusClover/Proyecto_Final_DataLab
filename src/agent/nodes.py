import logging
from src.agent.state import AgentState
from src.processors.document_loader import load_document
from src.processors.classifier import classify_document, create_vectorstore

logger = logging.getLogger(__name__)

def process_document_node(state: AgentState) -> AgentState:
    """Nodo Fase 1: Lee el documento"""
    file_path = state.get("file_path", "")
    logger.info(f"--- NODO INICIADO: Leyendo documento desde {file_path} ---")
    
    docs = load_document(file_path)
    
    if not docs:
        return {"documents": [], "status": "error_lectura"}
    
    file_type = file_path.split('.')[-1].lower()
    logger.info(f"--- NODO FINALIZADO: {len(docs)} fragmentos extraídos ---")
    
    return {
        "documents": docs, 
        "file_type": file_type,
        "status": "documento_cargado"
    }

def classify_node(state: AgentState) -> AgentState:
    """Nodo Fase 2: Clasifica el documento usando Gemini"""
    logger.info("--- NODO INICIADO: Clasificando documento ---")
    docs = state.get("documents", [])
    
    # Invocamos la función de tu compañero
    doc_class = classify_document(docs)
    
    logger.info(f"--- NODO FINALIZADO: Clasificación = {doc_class} ---")
    # Guardamos el resultado en la variable que acabamos de arreglar en state.py
    return {"document_class": doc_class, "status": "clasificado"}

def vectorize_document_node(state: AgentState) -> AgentState:
    """Nodo Fase 2: Crea la base de datos vectorial (Vectorstore)"""
    logger.info("--- NODO INICIADO: Vectorizando documentos ---")
    docs = state.get("documents", [])
    
    if not docs:
        logger.error("No hay documentos en el estado para vectorizar.")
        return {"status": "error_sin_documentos"}
    
    # Invocamos TU función
    exito = create_vectorstore(docs)
    
    if exito:
        logger.info("--- NODO FINALIZADO: Vectorización completada ---")
        return {"status": "vectorizado"}
    else:
        return {"status": "error_vectorizacion"}