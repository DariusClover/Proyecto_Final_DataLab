import os
import logging
from fpdf import FPDF

logger = logging.getLogger(__name__)

def limpiar_caracteres(texto: str) -> str:
    """
    Reemplaza acentos y caracteres especiales para evitar quiebres 
    de codificación en las fuentes base de FPDF2 dentro de Databricks.
    """
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "ü": "u", "Ü": "U"
    }
    for original, nuevo in reemplazos.items():
        texto = texto.replace(original, nuevo)
    return texto

def generate_pdf_report(state: dict) -> dict:
    """
    Nodo del grafo que toma la lista de análisis del estado,
    renderiza un reporte estructurado y guarda el archivo PDF final.
    """
    logger.info("--- NODO INICIADO: Generando Reporte PDF Consolidado ---")
    
    analyses = state.get("analyses", [])
    if not analyses:
        logger.error("Fallo de empaquetado: No se encontraron análisis en el estado.")
        state["status"] = "error_pdf"
        return state

    try:
        # Inicializar documento PDF limpio
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Encabezado Principal Corporativo
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "ConsulTech Analytics - Reporte Consolidado", ln=True, align="C")
        pdf.ln(4)
        
        # Línea divisoria
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        # Metadatos del lote de documentos
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Volumen de documentos procesados: {len(analyses)}", ln=True)
        pdf.cell(0, 6, "Entorno de ejecucion: Databricks Cloud Pipeline", ln=True)
        pdf.ln(10)

        # Sección de Contenido: Iteración sobre la lista de análisis
        for idx, item in enumerate(analyses):
            # Título del documento fuente
            pdf.set_font("Helvetica", "B", 12)
            nombre_archivo = limpiar_caracteres(item.get("file_name", "Desconocido"))
            pdf.cell(0, 8, f"[{idx + 1}] Archivo: {nombre_archivo}", ln=True)
            
            # Subtítulo con metadatos de clasificación
            pdf.set_font("Helvetica", "I", 9)
            clase = item.get("document_class", "N/A").upper()
            status_doc = item.get("status", "N/A")
            pdf.cell(0, 6, f"Clasificacion: {clase} | Estado del proceso: {status_doc}", ln=True)
            pdf.ln(4)
            
            # Cuerpo del Resumen Estructurado generado por Gemini
            pdf.set_font("Helvetica", "", 10)
            resumen_crudo = item.get("summary", "Sin contenido.")
            resumen_limpio = limpiar_caracteres(resumen_crudo)
            
            # Escribir bloque de texto con ajuste de línea automático
            pdf.multi_cell(0, 6, resumen_limpio)
            pdf.ln(8)
            
            # Separador entre registros
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(6)

        # Definir ruta de salida en la raíz del proyecto
        base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_final_path = os.path.join(output_dir, "reporte_final_consultech.pdf")
        
        # Exportar binario final
        pdf.output(pdf_final_path)
        
        logger.info(f"✅ Éxito de compilación: PDF guardado en {pdf_final_path}")
        state["pdf_path"] = pdf_final_path
        state["status"] = "finalizado"

    except Exception as e:
        logger.error(f"Fallo critico en la generacion del PDF. Error: {str(e)}")
        state["status"] = "error_pdf"

    return state