from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import analyze_document_batch_node, generate_pdf_report_node

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Definición de macro-nodos robustos
    workflow.add_node("analyze_batch", analyze_document_batch_node)
    workflow.add_node("generate_pdf", generate_pdf_report_node)
    
    # Flujo de ejecución lineal
    workflow.add_edge(START, "analyze_batch")
    workflow.add_edge("analyze_batch", "generate_pdf")
    workflow.add_edge("generate_pdf", END)
    
    return workflow.compile()