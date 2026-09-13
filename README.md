# 🤖 Agente el Maná — Industria de Alimentos el Maná

> Asistente de atención al cliente y empleados de Industria de Alimentos el Maná. Puedo ayudarte con información sobre 
  nuestras sedes, horarios, contacto, misión, visión, productos, precios, disponibilidad y descuentos. 
  También puedo consultar documentos internos (política comercial, protocolo, RRHH, plan estratégico) 
  y generar reportes ejecutivos o análisis de ventas.

## Descripción

- **Este proyecto implementa** un agente conversacional de IA para **Industria de Alimentos el Maná** que atiende de forma unificada  a 
     clientes  externos  y  empleados  internos.  Integra  técnicas  de memoria,  herramientas,  RAG  sobre  documentos corporativos  y 
	 orquestación multiagente (LangGraph) en un único servicio FastAPI desplegable. El objetivo es ofrecer respuestas  contextualizadas 
	 sobre productos, sedes, políticas internas y análisis de ventas, decidiendo autónomamente el camino más adecuado según la consulta.

## 🚀 Demo en vivo

- **Interfaz de chat:** `https://tu-proyecto.railway.app/`
- **API en producción:** `https://tu-proyecto.railway.app/api`
- **Documentación Swagger:** `https://tu-proyecto.railway.app/docs`
- **Estado del servicio:** `https://tu-proyecto.railway.app/health`

## 🧠 Capacidades del agente

- **Atención al cliente externo** (saludos, sedes, horarios, misión, visión, catálogo de productos, precios, combos y fallback amable para temas fuera de alcance).
- **Consultas internas para empleados** sobre política comercial, protocolo de atención, informe de RRHH y plan estratégico mediante RAG sobre PDFs corporativos.
- **Análisis multiagente** con cadena Investigador → Analista → Redactor para comparar, evaluar o investigar información de la empresa.
- **Reportes ejecutivos estructurados** (Pydantic) con resumen, alertas, oportunidades y recomendaciones priorizadas.
- **Memoria por sesión** para mantener contexto entre turnos y **uso de herramientas** (cálculos, fecha/hora, clima, búsqueda documental, análisis de ventas CSV).

## 🧭 Caminos de Ejecución

| Camino | Se activa cuando… |
|---|---|
| **`atencion_cliente`** | El usuario (cliente externo) saluda, pregunta por sedes, horarios, contacto, misión, visión, productos, precios, combos, o cualquier consulta general de la pastelería. Es el camino por defecto del coordinador. |
| **`documentos`** | El usuario (empleado interno) consulta sobre política comercial, protocolo de atención al cliente, informe de RRHH, plan estratégico u otra información contenida en los documentos internos (RAG sobre PDFs). |
| **`analisis`** | El usuario pide analizar, comparar, investigar o evaluar información de la empresa en profundidad (ej. "compara las ventas por región y saca conclusiones"). |
| **`reporte`** | El usuario solicita explícitamente un reporte, informe ejecutivo, resumen estructurado o "reporte de ventas", activando salida Pydantic validada. |

## 🔌 Endpoints Disponibles

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Sirve la interfaz de chat HTML (`frontend/index.html`) al abrir la app en el navegador. |
| `GET` | `/api` | Devuelve metadatos básicos de la API en JSON (nombre, versión, docs, health). |
| `GET` | `/health` | Verifica que el servicio está activo y responde con el nombre del agente y proveedor LLM. |
| `GET` | `/health/detail` | Estado detallado del servicio: agente, proveedor, modelo LLM y lista de herramientas disponibles. |
| `POST` | `/chat` | Envía un mensaje al agente y recibe su respuesta. Mantiene memoria por `session_id` e informa la categoría (camino) y fuentes RAG utilizadas. |
| `POST` | `/agent` | Ejecuta una tarea específica con contexto adicional opcional. Útil para consultas con instrucciones complejas o de dominio. |
| `DELETE` | `/chat/{session_id}` | Elimina el historial de conversación de una sesión, reiniciando el contexto del usuario. |
| `GET` | `/docs` | Documentación interactiva Swagger UI autogenerada por FastAPI. |
| `GET` | `/redoc` | Documentación ReDoc alternativa, también autogenerada por FastAPI. |

## 🛠️ Tecnologías

- **LLM:** Groq (openai/gpt-oss-120b)
- **Framework agente:** LangChain (`create_agent`, `with_structured_output`) + LangGraph (coordinador)
- **RAG:** búsqueda híbrida por palabras clave sobre `data/documentos/` (PDFs + CSV), sin dependencias pesadas de embeddings
- **API REST:** FastAPI + Uvicorn
- **Despliegue:** Railway
- **Lenguaje:** Python 3.11

## ⚡ Ejemplo de uso

```bash
# Conversar con el agente
curl -X POST https://tu-proyecto.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"mensaje": "Hola, ¿qué puedes hacer?", "session_id": "usuario_01"}'
```

```json
{
  "respuesta": "Hola! Soy [NOMBRE], especialista en [DOMINIO]...",
  "session_id": "usuario_01",
  "modelo": "openai/gpt-oss-120b"
}
```

## 🏃 Ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/InstructorJJ/proy_mana.git
cd tu-repo

# 2. Crear entorno virtual
conda create -n agentes python=3.11 -y
conda activate agentes

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves

# 5. Arrancar el servidor
uvicorn main:app --reload --port 8000

# 6. Ver documentación
# Abre http://localhost:8000/docs
```

## 🔧 Variables de entorno requeridas

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `GROQ_API_KEY` | Clave de API de Groq | ✅ |
| `GROQ_MODEL` | Modelo para conversación/RAG/análisis | ✅ |
| `GROQ_MODEL_ESTRUCTURADO` | Modelo para el nodo "reporte" (salida estructurada) | ✅ |
| `AGENT_NAME` | Nombre del agente | ✅ |
| `AGENT_ROLE` | Rol y especialización | ✅ |
| `LLM_PROVIDER` | `groq` o `ollama` | ✅ |
| `MEMORY_TYPE` | `buffer` o `window` | ✅ |
| `MEMORY_WINDOW_SIZE` | Mensajes a recordar si `MEMORY_TYPE=window` | Opcional |
| `DOCUMENTOS_DIR` | Carpeta con los PDFs/CSV del RAG | ✅ |
| `TAMANO_FRAGMENTO` / `SUPERPOSICION_FRAGMENTO` / `K_FRAGMENTOS` | Ventana de contexto del RAG | Opcional |
| `OPENWEATHER_API_KEY` | API de clima | Opcional |

## 📁 Estructura del proyecto

```
├── main.py                   # Servidor FastAPI — sirve el frontend y la API
├── s6_llm_factory.py         # Crea el LLM (Groq/Ollama, temperatura, modelo por nodo)
├── Procfile                  # Instrucción de arranque para Railway
├── railway.toml              # Configuración de Railway
├── requirements.txt          # Dependencias Python
├── .env.example              # Plantilla de variables de entorno
├── data/documentos/          # PDFs + CSV reales que lee el RAG
├── frontend/index.html       # Interfaz de chat (un solo archivo)
├── core/
│   ├── agent_core.py         # Orquesta el grafo + memoria por sesión
│   ├── atencion_cliente.py   # Sub-grafo multiagente de atención al cliente
│   ├── graph.py              # El coordinador LangGraph (4 caminos)
│   ├── rag.py                # Motor de RAG (búsqueda híbrida)
│   ├── tools.py              # Herramientas del agente
│   └── structured.py         # Esquemas Pydantic (salida estructurada)
├── models/
│   └── schemas.py            # Modelos Pydantic de la API
└── routes/
    ├── chat.py               # POST /chat, DELETE /chat/{session_id}
    ├── agent.py              # POST /agent
    └── health.py             # GET /health, /health/detail
```

## 👤 Autor

** Juan de Jesús Lizcano Sánchez **
- Seminario: Agentes de IA — Diseño, Integración y Despliegue en la Nube
- Universidad Santo Tomás - Maestría en Análisis de Datos y Sistemas Inteligentes

---

*Proyecto desarrollado como parte del seminario de Agentes de IA.*
# proy_mana
# proy_mana
