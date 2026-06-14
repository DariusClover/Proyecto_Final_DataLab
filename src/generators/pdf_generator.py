import os
import logging
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

logger = logging.getLogger(__name__)

def generate_pdf_report(analyses: list, template_name: str, output_path: str = "reporte_consolidado.pdf") -> bool:
    """
    Toma una lista de análisis (diccionarios), renderiza una plantilla HTML con Jinja2 
    y la convierte en un documento PDF profesional.
    """
    if not analyses:
        logger.warning("No hay análisis acumulados para generar el PDF.")
        return False

    try:
        # 1. Configurar Jinja2 para buscar en la carpeta 'templates'
        ruta_templates = os.path.join(os.getcwd(), 'templates')
        env = Environment(loader=FileSystemLoader(searchpath=ruta_templates))
        template = env.get_template(template_name)

        # 2. Inyectar los datos (analyses) en la plantilla HTML
        html_out = template.render(analyses=analyses)

        # 3. Guardar y convertir HTML a PDF usando xhtml2pdf
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, "w+b") as result_file:
            # pisa es el motor de xhtml2pdf
            pisa_status = pisa.CreatePDF(html_out, dest=result_file)

        if pisa_status.err:
            logger.error("Hubo errores al compilar el PDF con xhtml2pdf.")
            return False
            
        logger.info(f"ÉXITO: Reporte PDF generado profesionalmente en '{output_path}'")
        return True

    except Exception as e:
        logger.error(f"Fallo crítico en la generación del PDF: {str(e)}")
        return False