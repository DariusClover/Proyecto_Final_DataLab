from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    file_path: str                 # La ruta del archivo a leer
    documents: List[Document]      # Los textos extraídos por LangChain
    file_type: str                 # pdf, docx, txt
    document_class: str            # 'cientifico' o 'general'
    summary: str                   # El resumen generado por la Fase 3 (¡Nueva variable!)
    status: str                    # Para rastrear si hubo errores          # Para rastrear si hubo errores