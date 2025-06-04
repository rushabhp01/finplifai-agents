import azure.functions as func
import logging
import os
import pandas as pd
import json
import requests

import tempfile
import matplotlib
matplotlib.use('Agg')  # Usa backend sin GUI, ideal para entornos headless

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import uuid
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ToolCallResult,
    AgentStream,
    ReActAgent
)
from llama_index.llms.openai import OpenAI
from azure.core.exceptions import ResourceExistsError


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="agentGraficas", methods=["POST"])
async def agentGraficas (req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function para análisis de gráficas procesando la solicitud.')
    
    # Verificar API keys y configuraciones necesarias
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return func.HttpResponse(
            "Por favor, configura la variable de entorno OPENAI_API_KEY en la configuración de tu Function App.",
            status_code=500
        )
    
    # Obtener la cadena de conexión de Azure Blob Storage
    storage_connection_string = os.environ.get("AzureWebJobsStorage")
    if not storage_connection_string:
        return func.HttpResponse(
            "Por favor, configura la variable de entorno AzureWebJobsStorage en la configuración de tu Function App.",
            status_code=500
        )
    
    # Configuración del contenedor de Blob Storage
    blob_container = req.params.get('container', 'graficas-inditex-output')
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
        temp_file_path = os.path.join(temp_dir, "Cuentas.xlsx")
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        
        # Crear directorios de salida
        output_dir = os.path.join(tempfile.gettempdir(), "graficos_inditex")
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar los datos y generar gráficos/análisis
        result = await procesar_graficas_inditex(
            temp_file_path,
            output_dir,
            openai_api_key,
            storage_connection_string,
            blob_container,
            make_container_public,
            req.params.get('user_msg', "Por favor, genera un informe completo del balance de activos de la entidad en cuestion, incluyendo gráficos y análisis de ratios financieros.")
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

async def procesar_graficas_inditex(excel_path, output_dir, openai_api_key, storage_connection_string, blob_container, make_container_public, user_msg):
    """Procesa los datos de la empresa utilizando el agente LlamaIndex y sube resultados a Azure Blob Storage"""
    
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    
    try:
        container_client = blob_service_client.get_container_client(blob_container)

        if make_container_public:
            container_client.create_container(public_access=PublicAccess.BLOB)
            logging.info(f"Contenedor {blob_container} creado con acceso público")
        else:
            container_client.create_container()
            logging.info(f"Contenedor {blob_container} creado con acceso privado")

    except ResourceExistsError:
        logging.info(f"El contenedor {blob_container} ya existe")
    except Exception as e:
        logging.error(f"Error al crear el contenedor: {str(e)}")
    
        raise
    
    # Lista para almacenar información de archivos subidos
    uploaded_files = []
    execution_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    


    
    # Función para cargar y preparar los datos
    def cargar_y_preparar_datos():
        """Carga y prepara los datos financieros"""
        try:
            df = pd.read_excel(excel_path)
            
            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip().str.lower()
            
            # Convertir la columna 'activo' a string y rellenar NaN
            df['activo'] = df['activo'].fillna('').astype('string')
            
            # Encontrar índices de secciones importantes
            activo_no_corriente_inicio = df[df['activo'].str.contains('Activo no corriente', case=False, na=False)].index[0]
            activo_corriente_inicio = df[df['activo'].str.contains('Activo corriente', case=False, na=False)].index[0]
            pasivo_inicio = df[df['activo'].str.contains('pasivo', case=False, na=False)].index[0]
            
            # Extraer DataFrames específicos
            activo_no_corriente_df = df.iloc[activo_no_corriente_inicio + 1 : activo_corriente_inicio].copy()
            activo_corriente_df = df.iloc[activo_corriente_inicio + 1 : pasivo_inicio].copy()
            
            # Limpiar datos del activo no corriente
            activo_no_corriente_df = activo_no_corriente_df[~activo_no_corriente_df['activo'].str.strip().str.match(r'^[IVXLCDM]+\s')]
            activo_no_corriente_df['unnamed: 1'] = pd.to_numeric(activo_no_corriente_df['unnamed: 1'], errors='coerce')
            activo_no_corriente_df = activo_no_corriente_df.dropna(subset=['unnamed: 1'])
            activo_no_corriente_df = activo_no_corriente_df[abs(activo_no_corriente_df['unnamed: 1']) > 1]
            
            # Limpiar datos del activo corriente
            activo_corriente_df = activo_corriente_df[~activo_corriente_df['activo'].str.strip().str.match(r'^[IVXLCDM]+\s')]
            activo_corriente_df['unnamed: 1'] = pd.to_numeric(activo_corriente_df['unnamed: 1'], errors='coerce')
            activo_corriente_df = activo_corriente_df.dropna(subset=['unnamed: 1'])
            activo_corriente_df = activo_corriente_df[abs(activo_corriente_df['unnamed: 1']) > 1]
            
            # Acortar etiquetas para mejor visualización
            def acortar_etiqueta(etiqueta, max_length=40):
                return etiqueta.strip() if len(etiqueta) <= max_length else etiqueta.strip()[:max_length] + '...'
            
            activo_no_corriente_df['etiqueta_corta'] = activo_no_corriente_df['activo'].apply(acortar_etiqueta)
            activo_corriente_df['etiqueta_corta'] = activo_corriente_df['activo'].apply(acortar_etiqueta)
            
            return {
                "df_completo": df,
                "activo_no_corriente": activo_no_corriente_df,
                "activo_corriente": activo_corriente_df
            }
        except Exception as e:
            logging.error(f"Error al cargar y preparar datos: {str(e)}")
            return {"error": str(e)}

    # Cargar datos
    datos = cargar_y_preparar_datos()
    
    # Función para subir archivos a Blob Storage
    def subir_archivo(ruta_local, nombre_blob):
        try:
            blob_client = container_client.get_blob_client(nombre_blob)
            
            # Determinar el tipo de contenido basado en la extensión
            content_type = 'text/markdown'
            if ruta_local.lower().endswith('.png'):
                content_type = 'image/png'
            elif ruta_local.lower().endswith('.csv'):
                content_type = 'text/csv'
            
            with open(ruta_local, 'rb') as data:
                blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type)
                )
            
            blob_url = blob_client.url
            
            if not make_container_public:
                # Generar SAS token para este blob específico
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=blob_container,
                    blob_name=nombre_blob,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=24)
                )
                # Crear URL con token SAS
                sas_url = f"{blob_url}?{sas_token}"
            else:
                sas_url = blob_url
            
            file_info = {
                "execution_id": execution_id,
                "timestamp": timestamp,
                "nombre": nombre_blob,
                "url": sas_url,
                "content_type": content_type,
                "size_bytes": os.path.getsize(ruta_local)
            }
            
            uploaded_files.append(file_info)
            return file_info
            
        except Exception as e:
            logging.error(f"Error al subir archivo {ruta_local} a Blob Storage: {str(e)}")
            return {"error": str(e)}
    
    # Herramientas para el agente
    def visualizar_activo_no_corriente() -> str:
        """
        Genera y guarda un gráfico de barras horizontales del activo no corriente.
        """
        try:
            if "error" in datos:
                return f"Error al cargar datos: {datos['error']}"
            
            df = datos["activo_no_corriente"]
            
            # Ordenar datos por valor
            df_sorted = df.sort_values('unnamed: 1', ascending=True)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Crear barras horizontales
            bars = ax.barh(df_sorted['etiqueta_corta'], df_sorted['unnamed: 1'], 
                        color=sns.color_palette('viridis', len(df_sorted)))
            
            # Añadir valores a las barras
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:,.0f}',
                    va='center', fontsize=9)
            
            # Añadir título y etiquetas
            ax.set_title('Activo No Corriente', fontsize=16, pad=20)
            ax.set_xlabel('Valor (€)', fontsize=12)
            plt.tight_layout()
            
            # Guardar gráfico localmente
            #os.makedirs(output_dir, exist_ok=True)
            ruta_archivo = os.path.join(output_dir, "activo_no_corriente.png")
            plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Subir a Blob Storage
            blob_name = f"{timestamp}/{execution_id}/activo_no_corriente.png"
            info = subir_archivo(ruta_archivo, blob_name)
            
            return f"Gráfico del activo no corriente generado y subido a {info['url']}"
        except Exception as e:
            return f"Error al generar el gráfico: {str(e)}"

    def visualizar_activo_corriente() -> str:
        """
        Genera y guarda un gráfico de barras horizontales del activo corriente.
        """
        try:
            if "error" in datos:
                return f"Error al cargar datos: {datos['error']}"
            
            df = datos["activo_corriente"]
            
            # Ordenar datos por valor
            df_sorted = df.sort_values('unnamed: 1', ascending=True)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Crear barras horizontales
            bars = ax.barh(df_sorted['etiqueta_corta'], df_sorted['unnamed: 1'], 
                        color=sns.color_palette('mako', len(df_sorted)))
            
            # Añadir valores a las barras
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:,.0f}',
                    va='center', fontsize=9)
            
            # Añadir título y etiquetas
            ax.set_title('Activo Corriente', fontsize=16, pad=20)
            ax.set_xlabel('Valor (€)', fontsize=12)
            plt.tight_layout()
            
            # Guardar gráfico
            ruta_archivo = os.path.join(output_dir, "activo_corriente.png")
            plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Subir a Blob Storage
            blob_name = f"{timestamp}/{execution_id}/activo_corriente.png"
            info = subir_archivo(ruta_archivo, blob_name)
            
            return f"Gráfico del activo corriente generado y subido a {info['url']}"
        except Exception as e:
            return f"Error al generar el gráfico: {str(e)}"

    def generar_comparativa_activos() -> str:
        """
        Genera un gráfico comparativo entre activo corriente y no corriente y lo guarda.
        """
        try:
            if "error" in datos:
                return f"Error al cargar datos: {datos['error']}"
            
            total_no_corriente = datos["activo_no_corriente"]['unnamed: 1'].sum()
            total_corriente = datos["activo_corriente"]['unnamed: 1'].sum()
            
            labels = ['Activo No Corriente', 'Activo Corriente']
            values = [total_no_corriente, total_corriente]
            
            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, values, color=sns.color_palette('Set2'))
            
            # Añadir valores encima de las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=11)
            
            ax.set_title('Comparación de Activos', fontsize=16)
            ax.set_ylabel('Valor (€)', fontsize=12)
            plt.tight_layout()
            
            # Guardar gráfico
            ruta_archivo = os.path.join(output_dir, "comparativa_activos.png")
            plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Subir a Blob Storage
            blob_name = f"{timestamp}/{execution_id}/comparativa_activos.png"
            info_barras = subir_archivo(ruta_archivo, blob_name)
            
            # Gráfico de sectores
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                            textprops={'fontsize': 12}, colors=sns.color_palette('Set2'))
            
            ax.set_title('Distribución de Activos', fontsize=16)
            plt.tight_layout()
            
            # Guardar gráfico de sectores
            ruta_archivo_pie = os.path.join(output_dir, "distribucion_activos_pie.png")
            plt.savefig(ruta_archivo_pie, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Subir a Blob Storage
            blob_name_pie = f"{timestamp}/{execution_id}/distribucion_activos_pie.png"
            info_pie = subir_archivo(ruta_archivo_pie, blob_name_pie)
            
            porcentaje_no_corriente = (total_no_corriente / (total_no_corriente + total_corriente)) * 100
            porcentaje_corriente = (total_corriente / (total_no_corriente + total_corriente)) * 100
            
            return (f"Gráficos comparativos generados y subidos.\n"
                    f"Total Activo No Corriente: {total_no_corriente:,.2f}€ ({porcentaje_no_corriente:.1f}%)\n"
                    f"Total Activo Corriente: {total_corriente:,.2f}€ ({porcentaje_corriente:.1f}%)\n"
                    f"Gráfico de barras: {info_barras['url']}\n"
                    f"Gráfico de sectores: {info_pie['url']}")
        except Exception as e:
            return f"Error al generar los gráficos comparativos: {str(e)}"

    def generar_informe_completo() -> str:
        """
        Genera todos los gráficos disponibles y un informe básico del balance de activos.
        """
        try:
            if "error" in datos:
                return f"Error al cargar datos: {datos['error']}"
            
            # Generar todos los gráficos
            visualizar_activo_no_corriente()
            visualizar_activo_corriente()
            generar_comparativa_activos()
            
            # Crear informe
            total_no_corriente = datos["activo_no_corriente"]['unnamed: 1'].sum()
            total_corriente = datos["activo_corriente"]['unnamed: 1'].sum()
            total_activos = total_no_corriente + total_corriente
            
            # Principales componentes del activo no corriente
            no_corriente_principales = datos["activo_no_corriente"].nlargest(5, 'unnamed: 1')
            corriente_principales = datos["activo_corriente"].nlargest(5, 'unnamed: 1')
            
            # Crear informe en markdown
            informe_texto = f"""# Informe de Análisis de Activos

## Resumen General
- **Total Activos**: {total_activos:,.2f}€
- **Activo No Corriente**: {total_no_corriente:,.2f}€ ({(total_no_corriente/total_activos*100):.1f}%)
- **Activo Corriente**: {total_corriente:,.2f}€ ({(total_corriente/total_activos*100):.1f}%)

## Principales Componentes del Activo No Corriente
| Concepto | Valor (€) | % del Activo No Corriente |
|----------|-----------|---------------------------|
"""
            
            for _, row in no_corriente_principales.iterrows():
                porcentaje = (row['unnamed: 1'] / total_no_corriente) * 100
                informe_texto += f"| {row['activo']} | {row['unnamed: 1']:,.2f} | {porcentaje:.1f}% |\n"
            
            informe_texto += f"""
## Principales Componentes del Activo Corriente
| Concepto | Valor (€) | % del Activo Corriente |
|----------|-----------|------------------------|
"""
            
            for _, row in corriente_principales.iterrows():
                porcentaje = (row['unnamed: 1'] / total_corriente) * 100
                informe_texto += f"| {row['activo']} | {row['unnamed: 1']:,.2f} | {porcentaje:.1f}% |\n"
            
            informe_texto += f"""
## Conclusión
El análisis muestra una estructura de activos donde el {(total_no_corriente/total_activos*100):.1f}% corresponde a activos no corrientes y el {(total_corriente/total_activos*100):.1f}% a activos corrientes.

## Análisis Visual
Los gráficos generados se pueden encontrar en los siguientes enlaces:
- [Activo No Corriente](placeholder_url_activo_no_corriente)
- [Activo Corriente](placeholder_url_activo_corriente)
- [Comparativa de Activos](placeholder_url_comparativa)
- [Distribución de Activos](placeholder_url_distribucion)
"""
            
            # Guardar informe
            ruta_informe = os.path.join(output_dir, "informe_activos.md")
            with open(ruta_informe, "w", encoding="utf-8") as f:
                f.write(informe_texto)
            
            # Subir a Blob Storage
            blob_name_informe = f"{timestamp}/{execution_id}/informe_activos.md"
            info_informe = subir_archivo(ruta_informe, blob_name_informe)
            
            # Reemplazar los placeholder_url con las URL reales
            for file_info in uploaded_files:
                if "activo_no_corriente.png" in file_info["nombre"]:
                    informe_texto = informe_texto.replace("placeholder_url_activo_no_corriente", file_info["url"])
                elif "activo_corriente.png" in file_info["nombre"]:
                    informe_texto = informe_texto.replace("placeholder_url_activo_corriente", file_info["url"])
                elif "comparativa_activos.png" in file_info["nombre"]:
                    informe_texto = informe_texto.replace("placeholder_url_comparativa", file_info["url"])
                elif "distribucion_activos_pie.png" in file_info["nombre"]:
                    informe_texto = informe_texto.replace("placeholder_url_distribucion", file_info["url"])
            
            # Actualizar el informe con las URLs reales
            with open(ruta_informe, "w", encoding="utf-8") as f:
                f.write(informe_texto)
            
            # Subir la versión actualizada
            info_informe = subir_archivo(ruta_informe, blob_name_informe)
            
            return f"Informe completo y todos los gráficos generados con éxito. Informe disponible en: {info_informe['url']}"
        except Exception as e:
            return f"Error al generar el informe completo: {str(e)}"

    def analizar_ratios_financieros() -> str:
        """
        Analiza ratios financieros básicos relacionados con los activos.
        """
        try:
            if "error" in datos:
                return f"Error al cargar datos: {datos['error']}"
            
            total_no_corriente = datos["activo_no_corriente"]['unnamed: 1'].sum()
            total_corriente = datos["activo_corriente"]['unnamed: 1'].sum()
            
            # Ratio de estructura (Activo No Corriente / Activo Corriente)
            ratio_estructura = total_no_corriente / total_corriente if total_corriente != 0 else "No calculable"
            
            # Interpretaciones
            if isinstance(ratio_estructura, float):
                if ratio_estructura > 2:
                    interpretacion = "La empresa tiene una alta proporción de activos a largo plazo, lo que puede indicar una fuerte inversión en capacidad productiva o activos fijos."
                elif ratio_estructura > 1:
                    interpretacion = "La empresa tiene más recursos a largo plazo que a corto plazo, lo que es común en empresas industriales o con alta inversión en infraestructura."
                elif ratio_estructura > 0.5:
                    interpretacion = "La empresa mantiene un equilibrio entre activos a largo y corto plazo."
                else:
                    interpretacion = "La empresa tiene mayor proporción de activos corrientes, lo que puede indicar un enfoque en liquidez o un modelo de negocio con menos necesidad de activos fijos."
            else:
                interpretacion = "No se puede calcular la interpretación debido a valores inválidos."
                
            # Preparar resultado
            resultado = f"""## Análisis de Ratios Financieros

### Ratio de Estructura de Activos
- **Ratio Activo No Corriente / Activo Corriente**: {ratio_estructura if isinstance(ratio_estructura, str) else f'{ratio_estructura:.2f}'}

### Interpretación
{interpretacion}

### Composición del Activo
- **Porcentaje de Activo No Corriente**: {(total_no_corriente/(total_no_corriente+total_corriente)*100):.2f}%
- **Porcentaje de Activo Corriente**: {(total_corriente/(total_no_corriente+total_corriente)*100):.2f}%
"""
            
            # Guardar análisis
            ruta_analisis = os.path.join(output_dir, "analisis_ratios.md")
            with open(ruta_analisis, "w", encoding="utf-8") as f:
                f.write(resultado)
                
            # Subir a Blob Storage
            blob_name_analisis = f"{timestamp}/{execution_id}/analisis_ratios.md"
            info_analisis = subir_archivo(ruta_analisis, blob_name_analisis)
                
            return f"Análisis de ratios financieros completado y subido a {info_analisis['url']}"
        except Exception as e:
            return f"Error al analizar ratios financieros: {str(e)}"
    
    # Inicializar LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
    
    # Crear el agente
    finanzas_agent = ReActAgent(
        name="analisis_financiero",
        description="Analiza y visualiza datos financieros de una empresa, generando gráficos e informes personalizados.",
        system_prompt="""Actúa como un asistente experto en análisis financiero. 
        Utiliza las herramientas disponibles para generar visualizaciones y análisis de los datos financieros.
        Explica los resultados de forma clara y profesional, interpretando los datos cuando sea posible.""",
        tools=[
            visualizar_activo_no_corriente,
            visualizar_activo_corriente,
            generar_comparativa_activos,
            generar_informe_completo,
            analizar_ratios_financieros
        ],
        llm=llm
    )
    
    agent = AgentWorkflow(agents=[finanzas_agent], root_agent="analisis_financiero")
    
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
    


    markdown_content = ""
    for file_info in uploaded_files:
        if file_info["nombre"].endswith("informe_activos.md"):
            # Leer el contenido del archivo markdown
            with open(os.path.join(output_dir, "informe_activos.md"), "r", encoding="utf-8") as f:
                markdown_content = f.read()
                break
    
    # Leer también el contenido del análisis de ratios si existe
    ratios_content = ""
    for file_info in uploaded_files:
        if file_info["nombre"].endswith("analisis_ratios.md"):
            with open(os.path.join(output_dir, "analisis_ratios.md"), "r", encoding="utf-8") as f:
                ratios_content = f.read()
                break
    
    # Combinar ambos contenidos si existen
    if markdown_content and ratios_content:
        markdown_content += "\n\n" + ratios_content

    return {
        "agent_responses": responses,
        "tool_results": serializable_tool_results,
        "final_response": final_response_str,
        "files_generated": os.listdir(output_dir),
        "files_uploaded_to_blob": uploaded_files,
        "markdown_content": markdown_content
    }
