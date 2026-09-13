"""
core/atencion_cliente.py — Sub-grafo multiagente de atención al cliente
Industria de Alimentos el Mana · Integrado al despliegue FastAPI existente.

Las bases de conocimiento se cargan dinámicamente desde PDFs en
data/documentos/:

  KB_EMPRESA   ← 01_politica_comercial.pdf + 04_plan_estrategico.pdf
  KB_PRODUCTOS ← 05_catalogo_productos.pdf

Flujo:
  START → supervisor → greeting_worker → company_worker
        → products_worker → fallback_worker → aggregator → END
"""

import os
import operator
from pathlib import Path
from typing_extensions import TypedDict, Annotated

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END

from s6_llm_factory import crear_llm


# ═══════════════════════════════════════════════════════════
#  CARGA DE BASES DE CONOCIMIENTO DESDE PDFs
# ═══════════════════════════════════════════════════════════
DOCUMENTOS_DIR = Path(os.getenv("DOCUMENTOS_DIR", "data/documentos"))

PDFS_KB_EMPRESA = [
    os.getenv("PDF_KB_EMPRESA_1", "01_politica_comercial.pdf"),
    os.getenv("PDF_KB_EMPRESA_2", "04_plan_estrategico.pdf"),
]
PDF_KB_PRODUCTOS = os.getenv("PDF_KB_PRODUCTOS", "05_catalogo_productos.pdf")


def _cargar_pdf_texto(nombre_archivo: str) -> str:
    """Lee un PDF completo y devuelve su texto. Vacío si falla."""
    ruta = DOCUMENTOS_DIR / nombre_archivo
    if not ruta.exists():
        print(f"  ⚠️  [ATENCIÓN CLIENTE] No se encontró {ruta}")
        return ""
    try:
        paginas = PyPDFLoader(str(ruta)).load()
        texto = "\n\n".join(p.page_content for p in paginas if p.page_content.strip())
        print(f"  📄 [ATENCIÓN CLIENTE] KB cargada: {nombre_archivo} ({len(paginas)} pág.)")
        return texto
    except Exception as e:
        print(f"  ⚠️  [ATENCIÓN CLIENTE] Error leyendo {ruta}: {e}")
        return ""


def _cargar_multiples(nombres: list) -> str:
    """Carga varios PDFs y los concatena con un separador claro."""
    bloques = []
    for n in nombres:
        texto = _cargar_pdf_texto(n)
        if texto:
            bloques.append(f"### Documento: {n}\n\n{texto}")
    return "\n\n".join(bloques)


# Se ejecutan UNA sola vez al importar el módulo
KB_EMPRESA = _cargar_multiples(PDFS_KB_EMPRESA)
KB_PRODUCTOS = _cargar_pdf_texto(PDF_KB_PRODUCTOS)

if not KB_EMPRESA:
    KB_EMPRESA = "(No se pudo cargar la información de la empresa desde los PDFs.)"
if not KB_PRODUCTOS:
    KB_PRODUCTOS = "(No se pudo cargar el catálogo de productos desde el PDF.)"


# ═══════════════════════════════════════════════════════════
#  DOMINIOS SOPORTADOS (para el fallback)
# ═══════════════════════════════════════════════════════════
DOMINIOS_SOPORTADOS = """
- 👋 Saludos y conversación casual.
- 🏢 Información de la empresa: sedes, direcciones, misión, visión, horarios y contacto.
- 🛒 Productos: catálogo, precios, menú, disponibilidad y combos.
"""


# ═══════════════════════════════════════════════════════════
#  CONTRATO: Clasificación del supervisor
# ═══════════════════════════════════════════════════════════
class QuestionClassification(BaseModel):
    greeting: str = Field(default="", description="Saludo o conversación casual. Vacío si no hay.")
    company: str = Field(default="", description="Preguntas sobre la empresa. Vacío si no hay.")
    products: str = Field(default="", description="Preguntas sobre productos/precios. Vacío si no hay.")
    other: str = Field(
        default="",
        description=("Preguntas fuera de alcance (clima, deportes, política, "
                     "noticias, otras empresas, cultura general). Vacío si no hay."),
    )


# ═══════════════════════════════════════════════════════════
#  ESTADO COMPARTIDO
# ═══════════════════════════════════════════════════════════
class CustomerState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    greeting_question: str
    company_question: str
    products_question: str
    other_question: str
    greeting_response: str
    company_response: str
    products_response: str
    other_response: str
    final_response: str


# ═══════════════════════════════════════════════════════════
#  NODOS
# ═══════════════════════════════════════════════════════════
def supervisor(state: CustomerState) -> dict:
    user_msg = state["messages"][-1].content
    llm = crear_llm(temperature=0.0)
    classifier = llm.with_structured_output(QuestionClassification)

    try:
        result = classifier.invoke([
            SystemMessage(content=(
                "Eres el supervisor de atención al cliente de Industria de Alimentos el Mana.\n"
                "El cliente puede mandar UN solo mensaje con varias preguntas mezcladas.\n\n"
                "Clasifica cada parte del mensaje en:\n"
                "- greeting: saludos, despedidas, conversación casual\n"
                "- company: sedes, dirección, misión, visión, horarios, contacto\n"
                "- products: productos, precios, menú, disponibilidad, combos\n"
                "- other: TODO lo que no encaje (clima, deportes, política, otras empresas...)\n\n"
                "Copia el texto EXACTO de cada pregunta en su campo.\n"
                "Si una categoría no aplica, déjala vacía.\n"
                "IMPORTANTE: ante duda entre company/products y other, prefiere la específica."
            )),
            HumanMessage(content=f"Mensaje del cliente:\n{user_msg}"),
        ])
        return {
            "greeting_question": result.greeting,
            "company_question": result.company,
            "products_question": result.products,
            "other_question": result.other,
        }
    except Exception:
        return {
            "greeting_question": "",
            "company_question": user_msg,
            "products_question": "",
            "other_question": "",
        }


def greeting_worker(state: CustomerState) -> dict:
    question = state.get("greeting_question", "")
    if not question.strip():
        return {"greeting_response": ""}
    llm = crear_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente amable de Industria de Alimentos el Mana.\n"
            "Responde SOLO al saludo o conversación casual. Máximo 1-2 oraciones.\n"
            "NO respondas preguntas sobre productos o la empresa aquí."
        )),
        HumanMessage(content=question),
    ])
    return {"greeting_response": response.content}


def company_worker(state: CustomerState) -> dict:
    question = state.get("company_question", "")
    if not question.strip():
        return {"company_response": ""}
    llm = crear_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente de Industria de Alimentos el Mana.\n"
            "Responde SOLO con información de la base de conocimiento.\n"
            "Si no está en la base, di que no tienes esa información.\n\n"
            f"BASE DE CONOCIMIENTO (empresa):\n{KB_EMPRESA}"
        )),
        HumanMessage(content=question),
    ])
    return {"company_response": response.content}


def products_worker(state: CustomerState) -> dict:
    question = state.get("products_question", "")
    if not question.strip():
        return {"products_response": ""}
    llm = crear_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente de Industria de Alimentos el Mana.\n"
            "Responde SOLO con el catálogo de productos. Incluye precios exactos.\n"
            "Si el producto no está en el catálogo, dilo amablemente.\n\n"
            f"CATÁLOGO DE PRODUCTOS:\n{KB_PRODUCTOS}"
        )),
        HumanMessage(content=question),
    ])
    return {"products_response": response.content}


def fallback_worker(state: CustomerState) -> dict:
    question = state.get("other_question", "")
    if not question.strip():
        return {"other_response": ""}
    llm = crear_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente de Industria de Alimentos el Mana.\n\n"
            "El cliente preguntó algo FUERA de tu alcance.\n"
            "Debes:\n"
            "  1) Disculparte brevemente.\n"
            "  2) Aclarar que SÍ puedes ayudar con:\n"
            f"{DOMINIOS_SOPORTADOS}\n"
            "  3) Invitarlo a reformular.\n\n"
            "NO intentes responder la pregunta original. NO inventes datos.\n"
            "Máximo 3-4 oraciones, tono cálido."
        )),
        HumanMessage(content=f"Pregunta fuera de alcance: {question}"),
    ])
    return {"other_response": response.content}


def aggregator(state: CustomerState) -> dict:
    parts = []
    if state.get("greeting_response"):
        parts.append(f"[Saludo]\n{state['greeting_response']}")
    if state.get("company_response"):
        parts.append(f"[Info empresa]\n{state['company_response']}")
    if state.get("products_response"):
        parts.append(f"[Productos]\n{state['products_response']}")
    if state.get("other_response"):
        parts.append(f"[Fuera de alcance]\n{state['other_response']}")

    if not parts:
        return {"final_response": "Lo siento, no pude procesar tu consulta."}

    combined = "\n\n".join(parts)
    llm = crear_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "Eres el agente final de Industria de Alimentos el Mana.\n"
            "Combina las respuestas de los especialistas en UNA SOLA respuesta natural.\n"
            "Reglas:\n"
            "- Empieza con el saludo (si lo hay)\n"
            "- Responde cada tema en orden natural\n"
            "- Si hay '[Fuera de alcance]', intégrala al final con amabilidad\n"
            "- NO uses etiquetas como [Saludo] o [Productos]\n"
            "- Cierra ofreciendo ayuda adicional\n"
            "- Tono cálido, profesional, conciso"
        )),
        HumanMessage(content=(
            f"Mensaje original del cliente: {state['messages'][-1].content}\n\n"
            f"Respuestas de los especialistas:\n{combined}"
        )),
    ])
    return {"final_response": response.content}


# ═══════════════════════════════════════════════════════════
#  GRAFO
# ═══════════════════════════════════════════════════════════
def build_customer_agent():
    wf = StateGraph(CustomerState)
    wf.add_node("supervisor", supervisor)
    wf.add_node("greeting_worker", greeting_worker)
    wf.add_node("company_worker", company_worker)
    wf.add_node("products_worker", products_worker)
    wf.add_node("fallback_worker", fallback_worker)
    wf.add_node("aggregator", aggregator)

    wf.add_edge(START, "supervisor")
    wf.add_edge("supervisor", "greeting_worker")
    wf.add_edge("greeting_worker", "company_worker")
    wf.add_edge("company_worker", "products_worker")
    wf.add_edge("products_worker", "fallback_worker")
    wf.add_edge("fallback_worker", "aggregator")
    wf.add_edge("aggregator", END)
    return wf.compile()


customer_agent = build_customer_agent()


# ═══════════════════════════════════════════════════════════
#  API PÚBLICA
# ═══════════════════════════════════════════════════════════
def invoke_customer_agent(query: str) -> dict:
    """Invoca el sub-grafo multiagente y devuelve un dict listo para la API."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "greeting_question": "",
        "company_question": "",
        "products_question": "",
        "other_question": "",
        "greeting_response": "",
        "company_response": "",
        "products_response": "",
        "other_response": "",
        "final_response": "",
    }
    result = customer_agent.invoke(initial_state)
    return {
        "answer": result["final_response"],
        "classification": {
            "saludo": result["greeting_question"],
            "empresa": result["company_question"],
            "productos": result["products_question"],
            "fuera_de_alcance": result["other_question"],
        },
        "worker_responses": {
            "saludo": result["greeting_response"],
            "empresa": result["company_response"],
            "productos": result["products_response"],
            "fuera_de_alcance": result["other_response"],
        },
    }