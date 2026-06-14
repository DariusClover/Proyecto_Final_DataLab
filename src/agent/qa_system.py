import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

def setup_qa_chain(vectorstore_path: str = "vector_store/faiss_index"):
    """
    Carga la base de datos vectorial local (FAISS) y configura el motor de búsqueda 
    (Retrieval) conectado a Gemini para un sistema de Q&A robusto.
    """
    try:
        # 1. Cargar el mismo modelo de embeddings estadístico usado en la Fase 2
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Cargar FAISS desde el disco local
        # NOTA DE SEGURIDAD: allow_dangerous_deserialization=True es obligatorio en 
        # las nuevas versiones de LangChain al cargar índices FAISS locales.
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 3. Configurar el Retriever (El Buscador)
        # search_kwargs={"k": 3} indica que traerá los 3 fragmentos más relevantes del documento
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # 4. Inicializar Gemini con temperatura baja (0.2)
        # Esto asegura respuestas precisas y analíticas, minimizando alucinaciones.
        llm = get_llm(temperature=0.2)
        
        # 5. Construir la Cadena de Q&A Conversacional
        # Esta es la pieza clave que tu compañero usará en su notebook.
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True  # CRÍTICO: Requisito de la rúbrica para "citar fuentes"
        )
        
        logger.info("Motor de búsqueda Q&A y Retrieval inicializados correctamente.")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Fallo crítico al inicializar el motor de Q&A: {str(e)}")
        return None