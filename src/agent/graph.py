from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import process_document_node, classify_node, vectorize_document_node

def build_agent_graph():
    """
    Construye y compila el grafo principal de ConsulTech Analytics.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Agregar todos los Nodos de Fase 1 y 2
    workflow.add_node("loader", process_document_node)
    workflow.add_node("classifier", classify_node)
    workflow.add_node("vectorize", vectorize_document_node)
    
    # 2. Dibujar Aristas (Ruta lineal temporal)
    workflow.add_edge(START, "loader")
    workflow.add_edge("loader", "classifier")
    workflow.add_edge("classifier", "vectorize")
    
    # Por ahora termina aquí. En la Fase 3, el nodo 'classifier' 
    # enviará el flujo a los nodos de resumen usando un ConditionalEdge.
    workflow.add_edge("vectorize", END) 
    
    app = workflow.compile()
    
    return app