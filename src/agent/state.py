import operator
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document

class AgentState(TypedDict):
    file_path: str                 # La ruta del archivo actual
    documents: List[Document]      # Los textos extraídos por LangChain
    file_type: str                 # pdf, docx, txt
    document_class: str            # 'cientifico' o 'general'
    summary: str                   # El resumen del documento actual
    # ¡NUEVO!: Lista acumulativa de análisis para el reporte multi-documento
    analyses: Annotated[List[dict], operator.add] 
    status: str                    # Rastreo del estado