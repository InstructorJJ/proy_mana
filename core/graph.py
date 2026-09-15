"""
core/graph.py — Grafo coordinador (LangGraph)
Industria de Alimentos el Maná · Clientes + Empleados

      START → coordinador → [atencion_cliente | documentos | analisis | reporte] → END

  • atencion_cliente → sub-grafo multiagente para CLIENTES externos
  • documentos       → RAG para consultas INTERNAS / EMPLEADOS
  • analisis         → equipo Investigador → Analista → Redactor
  • reporte          → Salida Estructurada con Pydantic
"""

import os
import sys
from typing import TypedDict, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

from s6_llm_factory import crear_llm
from core.tools import TOOLS_BASE, buscar_documentos, analizar_ventas
from core.rag import obtener_motor
from core.structured import ReporteEjecutivo
from core.atencion_cliente import invoke_customer_agent as invoke_atencion_cliente

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "Agente el Maná")
ROL_AGENTE = os.getenv("AGENT_ROLE", "asistente de Industria de Alimentos el Maná")


def limpiar_pensamiento(texto: str) -> str:
    if texto and "</think>" in texto:
        return texto.split("</think>")[-1].strip()
    return (texto or "").strip()


# ═══════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════
class EstadoAgente(TypedDict):
    entrada_usuario: str
    historial: list
    categoria: str
    respuesta_final: str
    fuentes: list


# ═══════════════════════════════════════════════════════════
#  COORDINADOR
# ═══════════════════════════════════════════════════════════
def nodo_coordinador(estado: EstadoAgente) -> dict:
    llm = crear_llm(temperature=0.0)
    instrucciones = """Clasifica la pregunta del usuario en UNA categoría:

- atencion_cliente: consultas de CLIENTES EXTERNOS sobre Industria de
  Alimentos el Maná. Incluye saludos, conversación casual, sedes,
  direcciones, horarios, contacto, misión, visión, productos, precios,
  menú, disponibilidad, tortas, panes, bebidas y descuentos. Categoría por
  DEFECTO si dudas entre esta y otra.

- documentos: consultas INTERNAS / DE EMPLEADOS que requieren buscar en
  los documentos internos: política comercial, protocolo de atención,
  informe de RRHH y plan estratégico. TAMBIÉN incluye consultas FACTUALES
  de ventas (totales por sede/producto/mes, montos específicos, "cuánto
  se vendió", "ventas de X") que se responden con un dato directo.

- reporte: el usuario pide explícitamente un reporte, informe ejecutivo,
  resumen estructurado, o un "reporte de ventas".

- analisis: el usuario pide un análisis EN PROFUNDIDAD que requiera
  comparación, evaluación o conclusiones (ej. "compara las ventas por
  sede y saca conclusiones", "evalúa el desempeño del equipo de ventas").

Responde SOLO con la categoría, en minúsculas, sin explicación ni puntuación.

Ejemplos:
"Hola, ¿cómo estás?" -> atencion_cliente
"¿Cuál es la dirección de la sede de Bucaramanga?" -> atencion_cliente
"¿Cuánto vale la torta envinada?" -> atencion_cliente
"¿Qué horarios tienen los domingos?" -> atencion_cliente
"¿Cuál es la misión de la empresa?" -> atencion_cliente
"¿Tienen torta red velvet?" -> atencion_cliente
"¿Cuánto es 45 * 12?" -> atencion_cliente
"Ventas totales de Cabecera" -> documentos
"¿Cuánto vendió Bucaramanga?" -> documentos
"¿Cuál fue el producto más vendido?" -> documentos
"¿Cuánto se vendió en mayo?" -> documentos
"¿Cuál es el protocolo de atención al cliente?" -> documentos
"¿Cómo va la rotación de personal este trimestre?" -> documentos
"¿Qué dice la política comercial interna?" -> documentos
"¿Cuál es el plan estratégico 2025?" -> documentos
"Genera un reporte ejecutivo de ventas" -> reporte
"Necesito un informe ejecutivo de RRHH" -> reporte
"Analiza el desempeño de ventas por región y saca conclusiones" -> analisis
"Compara las ventas del Centro y Cabecera y saca conclusiones" -> analisis"""

    resp = llm.invoke([
        SystemMessage(content=instrucciones),
        HumanMessage(content=estado["entrada_usuario"]),
    ])
    contenido = limpiar_pensamiento(resp.content).lower()

    categoria = "atencion_cliente"
    # OJO: chequear "analisis" ANTES que "documentos" no es necesario aquí,
    # pero el orden importa si una palabra contiene a otra. Dejamos el orden
    # del más específico al más genérico.
    for opcion in ("analisis", "reporte", "documentos", "atencion_cliente"):
        if opcion in contenido:
            categoria = opcion
            break

    print(f"  🎯 [COORDINADOR] → {categoria.upper()}")
    return {"categoria": categoria}


def router(estado: EstadoAgente) -> Literal[
    "atencion_cliente", "documentos", "reporte", "analisis"
]:
    return estado.get("categoria", "atencion_cliente")


# ═══════════════════════════════════════════════════════════
#  NODO: ATENCIÓN AL CLIENTE — sub-grafo multiagente
# ═══════════════════════════════════════════════════════════
def nodo_atencion_cliente(estado: EstadoAgente) -> dict:
    print("  🏪 [ATENCIÓN CLIENTE] Activando sub-grafo multiagente...")
    resultado = invoke_atencion_cliente(estado["entrada_usuario"])
    return {
        "respuesta_final": resultado["answer"],
        "fuentes": [],
    }


# ═══════════════════════════════════════════════════════════
#  NODO: DOCUMENTOS — RAG sobre los PDFs/CSV internos
# ═══════════════════════════════════════════════════════════
def nodo_documentos(estado: EstadoAgente) -> dict:
    motor = obtener_motor()
    contexto, fuentes = motor.buscar(estado["entrada_usuario"])

    llm = crear_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres {NOMBRE_AGENTE}, asistente de Industria de
Alimentos el Maná. Responde ÚNICAMENTE basándote en el contexto de los
documentos internos proporcionado. Si la información no está en el contexto,
dilo claramente en vez de inventar. Cita el archivo/página cuando sea posible.
Responde en español, claro y profesional."""),
        ("human", "Contexto de los documentos internos:\n{contexto}\n\nPregunta: {pregunta}"),
    ])
    chain = prompt | llm | StrOutputParser()
    respuesta = chain.invoke({
        "contexto": contexto or "(no se encontró contexto relevante)",
        "pregunta": estado["entrada_usuario"],
    })
    return {"respuesta_final": limpiar_pensamiento(respuesta), "fuentes": fuentes}


# ═══════════════════════════════════════════════════════════
#  NODO: ANÁLISIS — equipo Investigador → Analista → Redactor
# ═══════════════════════════════════════════════════════════
def nodo_analisis(estado: EstadoAgente) -> dict:
    print("  🔍 [INVESTIGADOR] Recopilando información...")
    pregunta = estado["entrada_usuario"]

    # ── Pre-cómputo de herramientas (sin depender de tool calling) ──
    try:
        docs_ctx = buscar_documentos.invoke({"pregunta": pregunta})
    except Exception as e:
        docs_ctx = f"(no se pudo consultar documentos: {e})"

    try:
        ventas_region = analizar_ventas.invoke({"agrupar_por": "sede"})
        ventas_producto = analizar_ventas.invoke({"agrupar_por": "producto"})
        ventas_mes = analizar_ventas.invoke({"agrupar_por": "mes"})
        ventas_ctx = (
            f"{ventas_region}\n\n{ventas_producto}\n\n{ventas_mes}"
        )
    except Exception as e:
        ventas_ctx = f"(no se pudo consultar ventas: {e})"

    investigacion = limpiar_pensamiento(crear_llm(temperature=0.3).invoke([
        SystemMessage(content=(
            "Eres un investigador de Industria de Alimentos el Maná.\n"
            "Con base en el contexto proporcionado, extrae los datos relevantes "
            "a la pregunta del usuario. NO inventes cifras: si el dato no está "
            "en el contexto, dilo. Responde en español, en lista de puntos clave."
        )),
        HumanMessage(content=(
            f"Pregunta: {pregunta}\n\n"
            f"--- Documentos internos ---\n{docs_ctx}\n\n"
            f"--- Ventas por sede ---\n{ventas_region}\n\n"
            f"--- Ventas por producto ---\n{ventas_producto}\n\n"
            f"--- Ventas por mes ---\n{ventas_mes}"
        )),
    ]).content)

    print("  📊 [ANALISTA] Analizando información...")
    analisis = limpiar_pensamiento(crear_llm(temperature=0.4).invoke([
        SystemMessage(content=(
            "Eres un analista crítico de Industria de Alimentos el Maná. "
            "Extrae lo más importante, identifica riesgos u oportunidades, "
            "y prioriza los hallazgos. Responde en español, estructurado."
        )),
        HumanMessage(content=(
            f"Pregunta: {pregunta}\n\n"
            f"Información recopilada:\n{investigacion}\n\nAnaliza y concluye."
        )),
    ]).content)

    print("  ✍️  [REDACTOR] Redactando respuesta final...")
    redaccion = limpiar_pensamiento(crear_llm(temperature=0.5).invoke([
        SystemMessage(content=(
            f"Eres {NOMBRE_AGENTE}. Toma el análisis recibido y conviértelo "
            "en una respuesta final clara y bien organizada para el usuario. "
            "Responde en español."
        )),
        HumanMessage(content=(
            f"Pregunta original: {pregunta}\n\n"
            f"Análisis:\n{analisis}\n\nRedacta la respuesta final."
        )),
    ]).content)

    return {"respuesta_final": redaccion}


# ═══════════════════════════════════════════════════════════
#  NODO: REPORTE — Salida Estructurada con Pydantic
# ═══════════════════════════════════════════════════════════
def nodo_reporte(estado: EstadoAgente) -> dict:
    modelo_estructurado = os.getenv("GROQ_MODEL_ESTRUCTURADO", "openai/gpt-oss-120b")
    llm_estructurado = crear_llm(temperature=0.4, modelo=modelo_estructurado).with_structured_output(ReporteEjecutivo)
    datos_ventas = analizar_ventas.invoke({"agrupar_por": "sede"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Genera un reporte ejecutivo de Industria de Alimentos
el Maná basado en los datos disponibles. Sé específico y accionable.
Responde en español."""),
        ("human", "Solicitud: {solicitud}\n\nDatos disponibles:\n{datos}"),
    ])
    chain = prompt | llm_estructurado
    reporte: ReporteEjecutivo = chain.invoke({
        "solicitud": estado["entrada_usuario"],
        "datos": datos_ventas,
    })

    salida = f"📋 REPORTE EJECUTIVO — Industria de Alimentos el Maná\n\n{reporte.resumen_ejecutivo}\n"
    if reporte.alertas:
        salida += "\nALERTAS:\n"
        for a in reporte.alertas:
            salida += f"  • [{a.area}] {a.descripcion}\n    → {a.accion_recomendada}\n"
    if reporte.oportunidades:
        salida += "\nOPORTUNIDADES:\n" + "\n".join(f"  • {o}" for o in reporte.oportunidades) + "\n"
    if reporte.recomendaciones:
        salida += "\nRECOMENDACIONES:\n" + "\n".join(f"  • {r}" for r in reporte.recomendaciones)

    return {"respuesta_final": salida}


# ═══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO
# ═══════════════════════════════════════════════════════════
def construir_grafo():
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("coordinador", nodo_coordinador)
    grafo.add_node("atencion_cliente", nodo_atencion_cliente)
    grafo.add_node("documentos", nodo_documentos)
    grafo.add_node("analisis", nodo_analisis)
    grafo.add_node("reporte", nodo_reporte)

    grafo.add_edge(START, "coordinador")
    grafo.add_conditional_edges("coordinador", router, {
        "atencion_cliente": "atencion_cliente",
        "documentos": "documentos",
        "analisis": "analisis",
        "reporte": "reporte",
    })
    grafo.add_edge("atencion_cliente", END)
    grafo.add_edge("documentos", END)
    grafo.add_edge("analisis", END)
    grafo.add_edge("reporte", END)

    return grafo.compile()