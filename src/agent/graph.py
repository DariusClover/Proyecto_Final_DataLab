from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import process_document_node, classify_node, vectorize_document_node,summarize_node


def build_agent_graph():
    """
    Construye y compila el grafo principal de ConsulTech Analytics.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Agregar todos los Nodos de Fase 1, 2 y 3
    workflow.add_node("loader", process_document_node)
    workflow.add_node("classifier", classify_node)
    workflow.add_node("vectorize", vectorize_document_node)
    workflow.add_node("summarize", summarize_node)  # <-- Nuevo Nodo Registrado
    
    # 2. Dibujar Aristas (Conexión lineal del flujo)
    workflow.add_edge(START, "loader")
    workflow.add_edge("loader", "classifier")
    workflow.add_edge("classifier", "vectorize")
    workflow.add_edge("vectorize", "summarize")     # <-- Del vectorizador pasa al resumen
    workflow.add_edge("summarize", END)             # <-- El proceso termina temporalmente aquí
    
    app = workflow.compile()
    
    return app