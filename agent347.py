import azure.functions as func
import logging
import os
import pandas as pd
import json
import tempfile
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ToolCallResult,
    AgentStream,
    ReActAgent
)
from llama_index.llms.openai import OpenAI

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="agent347_func2")
async def modelo347_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
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
    
    blob_container = req.params.get('container', 'modelo347-output2')
    
    make_container_public = req.params.get('public_access', 'true').lower() == 'true'


    
    
    
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
        temp_file_path = os.path.join(temp_dir, "output.csv")
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)



        df = pd.read_csv(temp_file_path)

        
        
        
                
        output_dir = os.path.join(tempfile.gettempdir(), "salida_modelo_347")
        os.makedirs(output_dir)
        
        result = await process_modelo_347(
            df, 
            output_dir,
            openai_api_key, 
            storage_connection_string,
            blob_container,
            make_container_public,
            req.params.get('user_msg', "Procesa los datos para el modelo 347")
        )
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200
        )
        
        
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )

async def process_modelo_347(df, output_dir,openai_api_key, storage_connection_string, blob_container, make_container_public,  user_msg):
    """Process Model 347 data using LlamaIndex agent and upload results to Azure Blob Storage"""
    
    from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta
    import uuid
    
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
    
    def procesar_modelo_347() -> str:
        """
        Filtra proveedores con facturación total > 3.005,06 € y guarda un archivo CSV por proveedor
        en Azure Blob Storage.
        """
        import uuid
        import datetime
        from datetime import timedelta
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        
        a = df.groupby("nombre_proveedor", as_index=False)["total_factura"].sum()
        b = a[a.total_factura > 3005.06]
        proveedores = b.nombre_proveedor.tolist()
        df_filtrado = df[df['nombre_proveedor'].isin(proveedores)]
        df_agrupado = df_filtrado.groupby("nombre_proveedor")
        
        result_files = []
        
        execution_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        for nombre, grupo in df_agrupado:
            file_id = str(uuid.uuid4())[:8]
            
            nombre_archivo_local = f"{output_dir}/{nombre}_grupo.csv"
            grupo.to_csv(nombre_archivo_local, index=False)
            
            blob_name = f"{timestamp}/{execution_id}/{nombre}_{file_id}.csv"
            blob_client = container_client.get_blob_client(blob_name)
            
            with open(nombre_archivo_local, 'rb') as data:
                blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    content_settings=ContentSettings(content_type='text/csv')
                )
            
            blob_url = blob_client.url
            
            if not make_container_public:
                # Generar SAS token para este blob específico
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=blob_container,
                    blob_name=blob_name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    #expiry=datetime.datetime.utcnow() + timedelta(hours=sas_expiry_hours)
                )
                # Crear URL con token SAS
                sas_url = f"{blob_url}?{sas_token}"
            else:
                sas_url = blob_url
            
            file_info = {
                "id": file_id,
                "execution_id": execution_id,
                "timestamp": timestamp,
                "proveedor": nombre,
                "nombre": blob_name,
                "url": sas_url,  
                "content_type": "text/csv",
                "size_bytes": os.path.getsize(nombre_archivo_local)
            }
            
            uploaded_files.append(file_info)
            
            result_files.append({
                "id": file_id,
                "execution_id": execution_id,
                "proveedor": nombre,
                "total": float(b[b.nombre_proveedor == nombre].total_factura.values[0]),
                "num_facturas": len(grupo),
                "archivo_blob": sas_url  
            })
            
        return f"Generados y subidos a Azure Blob Storage {len(proveedores)} archivos para proveedores que superan los 3005.06 €. ID de ejecución: {execution_id}"
    

    def generate_markdown_response(final_response, uploaded_files):
        if not uploaded_files:
            return f"""
# 📊 Procesamiento Modelo 347

## ℹ️ Resultado
{final_response}

❌ **No se han generado archivos en esta operación.**

---
*Si esperaba obtener archivos, verifique los datos de entrada y los parámetros de filtrado.*
"""

        # Si hay archivos, generar resumen
        first_file = uploaded_files[0]
        execution_id = first_file.get("execution_id", "N/A")
        timestamp = first_file.get("timestamp", "N/A")

        # Formatear fecha si es posible
        try:
            formatted_date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
        except Exception:
            formatted_date = timestamp

        markdown = f"""
    # 📊 Procesamiento Modelo 347

    ## ✅ Procesamiento completado

    ### 📋 Resumen
    - **ID de ejecución:** `{execution_id}`
    - **Fecha y hora:** {formatted_date}
    - **Total archivos generados:** {len(uploaded_files)}

    ### 📁 Archivos generados
    | Proveedor | ID | Archivos | Enlace |
    |-----------|----|----------|--------|
    """

        # Añadir cada archivo a la tabla
        for file in uploaded_files:
            proveedor = file.get("proveedor", "N/A")
            file_id = file.get("id", "N/A")
            nombre = file.get("nombre", "").split("/")[-1] if file.get("nombre") else "N/A"
            url = file.get("url", "#")

            markdown += f"| {proveedor} | {file_id} | {nombre} | [Descargar CSV]({url}) |\n"

        markdown += """
    ---
    *Para más información sobre el Modelo 347, consulte la [web de la Agencia Tributaria](https://sede.agenciatributaria.gob.es/).*
    """
        return markdown    
    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
    
    # Create the agent
    modelo_347_agent = ReActAgent(
        name="modelo_347",
        description="Filtra proveedores/clientes con facturación > 3.005,06€ y genera archivos individuales por proveedor en Azure Blob Storage.",
        system_prompt="Actúa como un asistente experto en preparar datos para el Modelo 347. Usa la herramienta para ejecutar el proceso cuando el usuario lo indique.",
        tools=[procesar_modelo_347],
        llm=llm
    )
    
    agent = AgentWorkflow(agents=[modelo_347_agent], root_agent="modelo_347")
    
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
        serializable_tool_results.append(serializable_result)


    
    
    return {
        "agent_responses": responses,
        "tool_results": serializable_tool_results,
        "final_response": final_response_str,
        "files_uploaded_to_blob": uploaded_files,
        "markdown_content":generate_markdown_response(final_response_str, uploaded_files)
    }
