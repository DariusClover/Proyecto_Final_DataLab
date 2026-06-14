import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Cargar las variables ocultas del archivo .env
load_dotenv()

# Validar que la llave existe en el entorno
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("¡ALERTA! No se encontró GEMINI_API_KEY en el archivo .env")

def get_llm(temperature=0.0, usar_gemma=True):
    """
    Inicializa y retorna el modelo de IA.
    Permite alternar fácilmente entre Gemma 2 27B y Gemini 1.5 Flash.
    """
    # Selección dinámica del endpoint basado en tu preferencia
    if usar_gemma:
        nombre_modelo = "gemma-2-27b-it"
    else:
        nombre_modelo = "gemini-3.5-flash"
        
    logger.info(f"Conectando al "cerebro": Instanciando modelo {nombre_modelo} con temp={temperature}")
    
    return ChatGoogleGenerativeAI(
        model=nombre_modelo,
        temperature=temperature,
        google_api_key=api_key,
        max_output_tokens=2048 # Límite seguro para respuestas estructuradas
    )