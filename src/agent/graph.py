from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import process_document_node

def build_agent_graph():
    """
    Construye y compila el grafo principal de ConsulTech Analytics.
    """
    #  Inicializamos el lienzo pasándole nuestro "Portafolio" (AgentState)
    workflow = StateGraph(AgentState)
    
    #  Agregamos nuestra primera estación de trabajo (Nodo)
    # Le damos el nombre "loader" y le asignamos la función que acabamos de crear
    workflow.add_node("loader", process_document_node)
    
    #  Dibujamos las vías (Aristas)
    # El punto de entrada START va directo a nuestro nodo 'loader'
    workflow.add_edge(START, "loader")
    
    # Por ahora (Fase 1), después de cargar, el proceso termina.
    # En la Fase 2, cambiaremos END por el nodo de Clasificación.
    workflow.add_edge("loader", END)
    
    #  Compilamos el grafo para que sea ejecutable
    app = workflow.compile()
    
    return app