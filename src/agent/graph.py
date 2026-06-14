from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState

# Importamos todos los nodos, incluyendo el nuevo
from src.agent.nodes import (
    process_document_node, 
    classify_node, 
    vectorize_document_node, 
    summarize_node,
    generate_pdf_node # <-- Nueva importación
)

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Nodos
    workflow.add_node("loader", process_document_node)
    workflow.add_node("classifier", classify_node)
    workflow.add_node("vectorize", vectorize_document_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("pdf_generator", generate_pdf_node) # <-- Nuevo nodo
    
    # Aristas
    workflow.add_edge(START, "loader")
    workflow.add_edge("loader", "classifier")
    workflow.add_edge("classifier", "vectorize")
    workflow.add_edge("vectorize", "summarize")
    workflow.add_edge("summarize", "pdf_generator")       # <-- Del resumen pasa al PDF
    workflow.add_edge("pdf_generator", END)               # <-- Finaliza el flujo
    
    app = workflow.compile()
    
    return app