from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    file_paths: List[str]          # Lista con todas las rutas de archivos cargados
    analyses: List[dict]           # Colección de análisis individuales (nombre, clase, resumen)
    pdf_path: str                  # Ruta de salida del PDF consolidado
    status: str                    # Variable de control de estado del pipeline