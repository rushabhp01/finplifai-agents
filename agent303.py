import azure.functions as func
import logging
import os
import pandas as pd
import json
import tempfile
import matplotlib.pyplot as plt
import matplotlib
import requests
import uuid
import datetime
from datetime import timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess, generate_blob_sas, BlobSasPermissions
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ToolCallResult,
    AgentStream,
    ReActAgent
)
from llama_index.llms.openai import OpenAI

# Configurar matplotlib para trabajar sin UI
matplotlib.use('Agg')

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="modelo303_func")
async def modelo303_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request for Modelo 303.')
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return func.HttpResponse(
            "Please set the OPENAI_API_KEY environment variable in your Function App configuration.",
            status_code=500
        )
    
    # Get Azure Blob Storage connection string
    storage_connection_string = os.environ.get("AzureWebJobsStorage")
    if not storage_connection_string:
        return func.HttpResponse(
            "Please set the AzureWebJobsStorage environment variable in your Function App configuration.",
            status_code=500
        )
    
    blob_container = req.params.get('container', 'modelo303-output')
    make_container_public = req.params.get('public_access', 'true').lower() == 'true'
    return_only_urls = req.params.get('only_urls', 'false').lower() == 'true'
    
    try:
        # Recibir el archivo Excel
        #excel_file = req.files.get('files')
        req_body = req.get_json()
        excel_file = req_body.get('files')
        print(excel_file)
        
        # Obtener la URL desde el query o el cuerpo JSON
        if not excel_file:
            return func.HttpResponse(
                "Por favor, proporciona una URL pública del archivo Excel en el parámetro 'files'.",
                status_code=400
            )

        # Descargar el archivo desde la URL
        response = requests.get(excel_file)
        if response.status_code != 200:
            return func.HttpResponse(
                f"No se pudo descargar el archivo desde la URL proporcionada. Código de estado: {response.status_code}",
                status_code=400
            )

        # Guardar el archivo en un directorio temporal
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "Cuentas.xlsx")
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        
        # Crear directorios de salida
        output_dir = os.path.join(tempfile.gettempdir(), "graficos_inditex")
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar los datos y generar 303
        result = await process_modelo_303(
            temp_file_path, 
            openai_api_key, 
            storage_connection_string,
            blob_container,
            make_container_public,
            req.params.get('user_msg', "Procesa los datos para el modelo 303 y genera el informe en lenguaje Markdown.")
        )
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error procesando la solicitud: {str(e)}")
        return func.HttpResponse(
            f"Ha ocurrido un error: {str(e)}",
            status_code=500
        )
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )

async def process_modelo_303(excel_path,openai_api_key, storage_connection_string, blob_container, make_container_public, user_msg):
    """Process Model 303 data using LlamaIndex agent and upload results to Azure Blob Storage"""
    
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    
    try:
        container_client = blob_service_client.get_container_client(blob_container)
        
        if make_container_public:
            container_client.create_container(public_access=PublicAccess.BLOB)
            logging.info(f"Container {blob_container} created with public access")
        else:
            container_client.create_container()
            logging.info(f"Container {blob_container} created with private access")
            
    except Exception as e:
        logging.error(f"Error creating container: {str(e)}")
    
    uploaded_files = []
    
    def generar_informe_modelo_303() -> str:
        """
        Procesa el archivo Excel del Modelo 303 y genera un informe en markdown y gráficos,
        subiéndolos a Azure Blob Storage.
        
        Returns:
            str: Mensaje indicando que el informe ha sido generado con las URLs del informe
                 en markdown y el gráfico de IVA repercutido.
        """
        # Cargar las tres hojas del Excel
        datos_fiscales = pd.read_excel(excel_path, sheet_name="Datos Fiscales", index_col=0).squeeze()
        emitidas = pd.read_excel(excel_path, sheet_name="Facturas Emitidas")
        recibidas = pd.read_excel(excel_path, sheet_name="Facturas Recibidas")

        # Agrupar por tipo de IVA
        repercutido = emitidas.groupby("Tipo IVA (%)").agg({
            "Base Imponible (€)": "sum",
            "Cuota IVA (€)": "sum"
        }).reset_index()

        # Casillas
        casillas_repercutido = {
            21: {"base": 0.0, "cuota": 0.0},
            10: {"base": 0.0, "cuota": 0.0},
            4:  {"base": 0.0, "cuota": 0.0}
        }

        for _, row in repercutido.iterrows():
            tipo = row["Tipo IVA (%)"]
            if tipo in casillas_repercutido:
                casillas_repercutido[tipo]["base"] = row["Base Imponible (€)"]
                casillas_repercutido[tipo]["cuota"] = row["Cuota IVA (€)"]

        cuota_devengada_total = sum(v["cuota"] for v in casillas_repercutido.values())

        def calcular_deduccion(row):
            deducible = str(row["Deducible"]).strip().lower()
            cuota = row["Cuota IVA (€)"]
            if deducible in ["sí", "si", "yes", "true"]:
                return cuota
            elif "%" in deducible:
                try:
                    porcentaje = float(deducible.replace("%", "").strip()) / 100
                    return cuota * porcentaje
                except:
                    return 0.0
            else:
                return 0.0

        recibidas["Cuota Deducible"] = recibidas.apply(calcular_deduccion, axis=1)
        cuota_deducible_total = recibidas["Cuota Deducible"].sum()

        casilla_32 = cuota_devengada_total - cuota_deducible_total

        # Variables para el informe
        contribuyente = datos_fiscales["Nombre / Razón Social"]
        nif = datos_fiscales["NIF"]
        regimen = datos_fiscales["Régimen IVA"]
        caja = datos_fiscales["Criterio de Caja"]
        devolucion = datos_fiscales["Solicita Devolución"]
        periodo = f"{datos_fiscales['Trimestre']} {datos_fiscales['Año']}"

        base_21 = casillas_repercutido[21]["base"]
        cuota_21 = casillas_repercutido[21]["cuota"]
        base_10 = casillas_repercutido[10]["base"]
        cuota_10 = casillas_repercutido[10]["cuota"]
        base_4 = casillas_repercutido[4]["base"]
        cuota_4 = casillas_repercutido[4]["cuota"]

        casilla_27 = cuota_devengada_total
        casilla_28 = cuota_deducible_total
        casilla_31 = cuota_deducible_total
        casilla_32 = casilla_27 - casilla_31

        # Para tabla de deducciones individuales
        tabla_deducciones = ""
        for _, row in recibidas.iterrows():
            concepto = row["Concepto"]
            cuota = row["Cuota IVA (€)"]
            deducible = str(row["Deducible"])
            cuota_deducida = calcular_deduccion(row)
            tabla_deducciones += f"| {concepto:<20} | {cuota:6.2f} | {deducible:>7} | {cuota_deducida:6.2f} |\n"

        # Markdown final
        markdown = f"""
# 🧾 Informe Modelo 303 – {periodo}

**Contribuyente:** {contribuyente}  
**NIF:** {nif}  
**Régimen IVA:** {regimen}  
**Criterio de Caja:** {caja}  
**Solicita devolución:** {devolucion}  

---

## 📊 IVA Repercutido (Facturas Emitidas)

| Tipo IVA | Casilla | Base Imponible (€) | Cuota IVA (€) |
|----------|---------|--------------------|----------------|
| 21%      | 01 / 02 | {base_21:,.2f}      | {cuota_21:,.2f} |
| 10%      | 03 / 04 | {base_10:,.2f}      | {cuota_10:,.2f} |
| 4%       | 05 / 06 | {base_4:,.2f}       | {cuota_4:,.2f}  |

**➡️ Casilla 27 (Total IVA Devengado):** `{casilla_27:,.2f} €`

---

## 📥 IVA Soportado Deducible (Facturas Recibidas)

| Concepto             | Cuota IVA (€) | % Deducible | Cuota Deducible (€) |
|----------------------|---------------|-------------|----------------------|
{tabla_deducciones.strip()}

**➡️ Casilla 28 (Total Deducible):** `{casilla_28:,.2f} €`  
**➡️ Casilla 31 (Total Deducciones):** `{casilla_31:,.2f} €`

---

## 💰 Resultado de la liquidación

| Descripción                  | Importe (€) |
|------------------------------|-------------|
| Cuota IVA Devengado (27)     | {casilla_27:,.2f}      |
| Cuota IVA Deducible (31)     | {casilla_31:,.2f}      |
| **➡️ Resultado (Casilla 32)**| **{casilla_32:,.2f}**  |

---

## ✅ Casillas a rellenar:  
`01`, `02`, `03`, `04`, `27`, `28`, `31`, `32`

📅 *Recuerda: este modelo debe presentarse antes del 20 del mes siguiente al trimestre.*
"""

        # Guardar markdown localmente
        markdown_file = os.path.join(tempfile.gettempdir(),"informe_modelo_303.md")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown)
            
        # Crear gráfico de IVA repercutido
        plt.figure(figsize=(10, 6))
        tipos_iva = ['21%', '10%', '4%']
        bases = [base_21, base_10, base_4]
        cuotas = [cuota_21, cuota_10, cuota_4]
        
        x = range(len(tipos_iva))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, bases, width, label='Base Imponible')
        ax.bar([i + width for i in x], cuotas, width, label='Cuota IVA')
        
        ax.set_ylabel('Euros')
        ax.set_title('IVA Repercutido por Tipo')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(tipos_iva)
        ax.legend()
        
        # Guardar gráfico localmente
        chart_file = os.path.join(tempfile.gettempdir(), "iva_repercutido_chart.png")
        plt.savefig(chart_file)
        plt.close()
        
        # Crear un ID de ejecución único
        execution_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Subir archivos al blob storage
        uploaded_files = []
        
        # Subir markdown
        md_blob_name = f"{timestamp}/{execution_id}/informe_modelo_303.md"
        md_blob_client = container_client.get_blob_client(md_blob_name)
        
        with open(markdown_file, 'rb') as data:
            md_blob_client.upload_blob(
                data, 
                overwrite=True,
                content_settings=ContentSettings(content_type='text/markdown')
            )
        
        md_blob_url = md_blob_client.url
        
        if not make_container_public:
            # Generar SAS token para este blob específico
            md_sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=blob_container,
                blob_name=md_blob_name,
                account_key=blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.datetime.utcnow() + timedelta(hours=24)
            )
            # Crear URL con token SAS
            md_sas_url = f"{md_blob_url}?{md_sas_token}"
        else:
            md_sas_url = md_blob_url
        
        # Agregar información del archivo markdown subido
        file_info = {
            "id": f"md_{str(uuid.uuid4())[:8]}",
            "execution_id": execution_id,
            "timestamp": timestamp,
            "nombre": md_blob_name,
            "url": md_sas_url,
            "type": "markdown",
            "content_type": "text/markdown",
            "size_bytes": os.path.getsize(markdown_file)
        }
        
        uploaded_files.append(file_info)
        
        # Subir gráfico
        chart_blob_name = f"{timestamp}/{execution_id}/iva_repercutido_chart.png"
        chart_blob_client = container_client.get_blob_client(chart_blob_name)
        
        with open(chart_file, 'rb') as data:
            chart_blob_client.upload_blob(
                data, 
                overwrite=True,
                content_settings=ContentSettings(content_type='image/png')
            )
        
        chart_blob_url = chart_blob_client.url
        
        if not make_container_public:
            # Generar SAS token para este blob específico
            chart_sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=blob_container,
                blob_name=chart_blob_name,
                account_key=blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.datetime.utcnow() + timedelta(hours=24)
            )
            # Crear URL con token SAS
            chart_sas_url = f"{chart_blob_url}?{chart_sas_token}"
        else:
            chart_sas_url = chart_blob_url
        
        # Agregar información del archivo chart subido
        file_info = {
            "id": f"chart_{str(uuid.uuid4())[:8]}",
            "execution_id": execution_id,
            "timestamp": timestamp,
            "nombre": chart_blob_name,
            "url": chart_sas_url,
            "type": "image",
            "content_type": "image/png",
            "size_bytes": os.path.getsize(chart_file)
        }
        
        uploaded_files.append(file_info)
        
        # Crear una respuesta más informativa con las URLs
        return (f"{markdown}\n\n"
            f"Informe del Modelo 303 generado y subido a Azure Blob Storage.\n\n"
            f"ID de ejecución: {execution_id}\n\n"
            f"URL del informe Markdown: {md_sas_url}\n\n"
            f"URL del gráfico de IVA repercutido: {chart_sas_url}"
        )
    
    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
    
    # Create the agent
    modelo_303_agent = ReActAgent(
        name="modelo_303",
        description="Procesa datos del Modelo 303 de IVA y genera un informe con gráficos.",
        system_prompt="Actúa como un asistente experto en preparar el Modelo 303 de IVA. Usa la herramienta para procesar el archivo Excel y generar un informe detallado cuando el usuario lo indique.",
        tools=[generar_informe_modelo_303],
        llm=llm
    )
    
    agent = AgentWorkflow(agents=[modelo_303_agent], root_agent="modelo_303")
    
    responses = []
    tool_results = []
    
    handler = agent.run(user_msg=user_msg)
    
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            tool_result = {
                "tool_name": ev.tool_name if hasattr(ev, 'tool_name') else "",
                "tool_kwargs": str(ev.tool_kwargs) if hasattr(ev, 'tool_kwargs') else "{}",
                "tool_output": str(ev.tool_output) if hasattr(ev, 'tool_output') else ""
            }
            tool_results.append(tool_result)
            logging.info(f"Tool called: {tool_result['tool_name']} => {tool_result['tool_output']}")
        elif isinstance(ev, AgentStream):
            delta = str(ev.delta) if hasattr(ev, 'delta') else ""
            responses.append(delta)
            logging.info(f"Agent response: {delta}")
    
    final_response = await handler
    
    final_response_str = str(final_response) if final_response else ""
    
    serializable_tool_results = []
    for result in tool_results:
        serializable_result = {
            "tool_name": result.get("tool_name", ""),
            "tool_kwargs": str(result.get("tool_kwargs", {})),
            "tool_output": str(result.get("tool_output", ""))
        }
        mark3 = str(result.get("tool_output", ""))
        serializable_tool_results.append(serializable_result)
    
    # Extraer URLs de los archivos para incluirlas claramente en la respuesta
    markdown_url = None
    chart_url = None
    
    for file in uploaded_files:
        if file.get("nombre", "").endswith(".md"):
            markdown_url = file.get("url")
        elif file.get("nombre", "").endswith(".png"):
            chart_url = file.get("url")
    
    return {
        "agent_responses": responses,
        "tool_results": serializable_tool_results,
        "markdown_content2": final_response_str,
        "markdown_content": mark3,

        "files_uploaded_to_blob": uploaded_files,
        "informe_markdown_url": markdown_url,
        "grafico_iva_url": chart_url,
        "urls": {
            "markdown": markdown_url,
            "chart": chart_url
        }
    }
