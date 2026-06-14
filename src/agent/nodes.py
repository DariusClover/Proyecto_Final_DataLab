import logging
from src.agent.state import AgentState
from src.processors.document_loader import load_document

logger = logging.getLogger(__name__)

def process_document_node(state: AgentState) -> AgentState:
    """
    Nodo inicial de la Fase 1.
    Toma la ruta del archivo del estado, extrae el texto y lo guarda.
    """
    file_path = state.get("file_path", "")
    logger.info(f"--- NODO INICIADO: Leyendo documento desde {file_path} ---")
    
    # Usamos la función de nuestros "sentidos" que creamos antes
    docs = load_document(file_path)
    
    if not docs:
        # Si hubo un error en la lectura, actualizamos el estado
        return {"documents": [], "status": "error_lectura"}
    
    # Extraemos el tipo de archivo de la ruta para guardarlo en la memoria
    file_type = file_path.split('.')[-1].lower()
    
    logger.info(f"--- NODO FINALIZADO: {len(docs)} fragmentos extraídos ---")
    
    # Devolvemos las variables que queremos actualizar en el Estado
    return {
        "documents": docs, 
        "file_type": file_type,
        "status": "documento_cargado"
    }