import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar las variables ocultas del archivo .env
load_dotenv()

# Validar que la llave existe en el entorno
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("¡ALERTA! No se encontró GEMINI_API_KEY en el archivo .env")

def get_llm(temperature=0.0):
    """
    Inicializa y retorna el modelo de Gemini.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=temperature,
        google_api_key=api_key
    )