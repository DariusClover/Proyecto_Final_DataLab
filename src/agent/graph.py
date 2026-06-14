from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import process_document_node, vectorize_document_node

def build_agent_graph():
    """
    Construye y compila el grafo principal de ConsulTech Analytics.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Agregar Nodos
    workflow.add_node("loader", process_document_node)
    workflow.add_node("vectorize", vectorize_document_node)
    
    # 2. Dibujar Aristas (Conexiones)
    workflow.add_edge(START, "loader")
    workflow.add_edge("loader", "vectorize")  # Del loader pasa a vectorizar
    
    # Por ahora termina aquí. Cuando tu compañero haga la clasificación, 
    # cambiaremos este END por un ConditionalEdge (Arista Condicional).
    workflow.add_edge("vectorize", END) 
    
    app = workflow.compile()
    
    return app