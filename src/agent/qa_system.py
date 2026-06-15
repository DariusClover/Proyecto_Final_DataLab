import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

def setup_qa_chain(vectorstore_path: str = "vector_store/faiss_index"):
    """
    Carga la base de datos vectorial local (FAISS) y configura el motor de búsqueda 
    (Retrieval) conectado al LLM con prompts personalizados y citación de fuentes.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Mantener consistencia con el modelo pulido y deshabilitar Gemma si corresponde
        llm = get_llm(temperature=0.2, usar_gemma=False)
        
        # DEFINICIÓN DEL PROMPT CORPORATIVO (Requisito de la rúbrica)
        template = (
            "Actúa como un Consultor Senior experto de ConsulTech Analytics.\n"
            "Tu objetivo es responder a la pregunta del usuario utilizando estrictamente la información "
            "encontrada en el 'Contexto del documento'. No inventes datos. Si la respuesta no está en el contexto, "
            "indica amablemente que el documento no contiene esa información.\n\n"
            "Contexto del documento:\n{context}\n\n"
            "Pregunta: {question}\n\n"
            "Respuesta del Consultor Senior:"
        )
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Construcción de la cadena inyectando el prompt estructurado
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT} # Enlace obligatorio
        )
        
        logger.info("Motor de búsqueda Q&A y Retrieval inicializados correctamente con Prompt Corporativo.")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Fallo crítico al inicializar el motor de Q&A: {str(e)}")
        return None