import logging
import azure.functions as func
import requests
import csv
import io
import datetime
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="agent-subvenciones", methods=["POST"])
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # ---------- CONFIGURACIÓN ----------
        IPYME_URL = "https://wapis.ipyme.org/servicioayudas/ayudas/ayudas"
        DETALLE_URL_BASE = "https://wapis.ipyme.org/servicioayudas/ayudas/detalle"
        HEADERS = {
            "User-Agent": "Mozilla/5.0"
        }
        
        # Parámetros de la petición (pueden venir del request o usar valores por defecto)
        texto_busqueda = req.params.get('query', 'comercio')
        en_plazo = req.params.get('enPlazo', 'true')
        ventana = int(req.params.get('ventana', '10000000'))
        pagina = int(req.params.get('pagina', '0'))
        
        # Obtener configuración del blob storage desde la configuración de la aplicación
        connect_str = os.environ['AzureWebJobsStorage']
        container_name = os.environ.get('BLOB_CONTAINER_NAME', 'ipyme-data')
        blob_base_url = os.environ.get('BLOB_BASE_URL', f"https://{os.environ.get('STORAGE_ACCOUNT_NAME', 'storage')}.blob.core.windows.net/{container_name}")
        
        # -----------------------------------
        # 1. Consulta API IPYME
        params_ipyme = {
            "texto": texto_busqueda,
            "enPlazo": en_plazo,
            "pagina": pagina,
            "ventana": ventana
        }
        
        logging.info(f"Consultando API IPYME con parámetros: {params_ipyme}")
        resp_ipyme = requests.get(IPYME_URL, params=params_ipyme, headers=HEADERS)
        
        # Validar respuesta
        if resp_ipyme.status_code != 200:
            error_msg = f"Error al consultar la API IPYME: {resp_ipyme.status_code} - {resp_ipyme.text}"
            markdown_response = generate_markdown_error(error_msg)
            return func.HttpResponse(
                markdown_response,
                status_code=500,
                mimetype="text/markdown"
            )
            
        ayudas_ipyme = resp_ipyme.json()
        logging.info(f"Se encontraron {len(ayudas_ipyme)} ayudas")
        
        # 2. Crear nombre del archivo con fecha y hora actual
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        blob_name = f"ayudas_ipyme_detalles_{timestamp}.csv"
        
        # 3. Crear CSV en memoria
        output = io.StringIO()
        campos = ["Id", "Titulo", "TituloCorto", "Ambito", "FechaSolicitud", "Vigentes", "EnlaceDetalle"]
        
        writer = csv.DictWriter(output, fieldnames=campos)
        writer.writeheader()
        
        for ayuda in ayudas_ipyme:
            id_ayuda = ayuda.get("Id", "")
            enlace_detalle = f"{DETALLE_URL_BASE}?id={id_ayuda}&fichero="
            fila = {
                "Id": id_ayuda,
                "Titulo": ayuda.get("Titulo", ""),
                "TituloCorto": ayuda.get("TituloCorto", ""),
                "Ambito": ayuda.get("Ambito", ""),
                "FechaSolicitud": ayuda.get("FechaSolicitud", ""),
                "Vigentes": ayuda.get("Vigentes", ""),
                "EnlaceDetalle": enlace_detalle
            }
            writer.writerow(fila)
            
        # 4. Guardar CSV en blob storage
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Asegurar que el contenedor existe
        try:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.get_container_properties()  # Verificar si existe
        except Exception as e:
            logging.info(f"Creando contenedor {container_name}")
            container_client = blob_service_client.create_container(container_name)
        
        # Subir el archivo al blob storage
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        output.seek(0)  # Volver al inicio del archivo en memoria
        blob_client.upload_blob(output.getvalue(), overwrite=True)
        
        # Construir URL completa del blob
        blob_url = f"{blob_base_url}/{blob_name}"
        
        logging.info(f"Archivo generado y subido: {blob_url}")
        
        # 5. Generar y devolver respuesta en formato Markdown
        markdown_response = generate_markdown_success(
            texto_busqueda=texto_busqueda, 
            en_plazo=(en_plazo.lower() == "true"), 
            total_ayudas=len(ayudas_ipyme),
            blob_name=blob_name,
            blob_url=blob_url,
            timestamp=now.strftime("%d/%m/%Y %H:%M:%S")
        )
        
        # Crear objeto JSON con el campo markdown_response
        json_response = {
            "markdown_content": markdown_response
        }
        
        return func.HttpResponse(
            json.dumps(json_response),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error en la función: {str(e)}")
        markdown_response = generate_markdown_error(f"Error en la ejecución de la función: {str(e)}")
        # Crear objeto JSON con el campo markdown_response
        json_response = {
            "markdown_content": markdown_response
        }
        
        return func.HttpResponse(
            json.dumps(json_response),
            status_code=500,
            mimetype="application/json"
        )

def generate_markdown_success(texto_busqueda, en_plazo, total_ayudas, blob_name, blob_url, timestamp):
    """Genera una respuesta en Markdown visualmente atractiva para casos exitosos"""
    
    estado_plazo = "en plazo activo" if en_plazo else "sin restricción de plazo"
    
    markdown = f"""
# 🎯 Consulta de Ayudas y Subvenciones

## ✅ Consulta completada con éxito

### 📊 Resumen de la consulta
- **Término de búsqueda:** "{texto_busqueda}"
- **Filtro de plazo:** {estado_plazo}
- **Total de ayudas encontradas:** {total_ayudas}

### 📁 Archivo generado
Se ha generado y subido correctamente un archivo CSV con los resultados:

📋 **Nombre del archivo:** `{blob_name}`

🔗 **Enlace de descarga:**  
[Descargar archivo CSV]({blob_url})

---
*Para más información sobre las ayudas disponibles, consulte el [Portal de Ayudas IPYME](https://www.ipyme.org/).*
"""
    return markdown

def generate_markdown_error(error_message):
    """Genera una respuesta en Markdown para casos de error"""
    
    markdown = f"""
# ❌ Error en la consulta de ayudas

## 🚨 No se ha podido completar la operación

### 📋 Detalles del error
```
{error_message}
```

### 🔄 Recomendaciones
- Verifica los parámetros de la consulta
- Asegúrate de que la API IPYME esté disponible
- Comprueba la configuración de almacenamiento

---
*Si el problema persiste, contacte con el administrador del sistema.*
"""
    return markdown
