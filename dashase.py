import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
from datetime import datetime
import numpy as np
from streamlit_option_menu import option_menu
import base64

# ML imports
from prophet import Prophet
from prophet.plot import plot_plotly
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Dashboard Ventas Interactivo", layout="wide")

# ====== PREMIUM THEME (Auto-injected) ======
custom_css = """
<style>
:root {
  --text: #e6edf3;
  --bg: #0b1324;
  --panel: rgba(255,255,255,0.08);
  --accent: #0a1856;
  --accent-2: #ffb869;
}
/* Fondo general */
[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Hero corporativo (estilo base) */
.hero {
  display:flex; justify-content:space-between; align-items:center;
  padding:25px 40px; border-radius:20px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 10px 34px rgba(0,0,0,0.35);
  margin-bottom:30px;
  color:white;
}
.hero h1 { font-size:36px; font-weight:900; margin:0; }
.hero h2 { font-size:18px; font-weight:500; margin-top:5px; }
.hero img { width:200px; border-radius:14px; box-shadow:0 6px 22px rgba(0,0,0,0.45); }

/* ✅ Fuerza texto BLANCO para el header renderizado vía st.markdown */
.asecom-hero, .asecom-hero * { color:#ffffff !important; }

/* Tarjetas glass */
.card-glass {
  background: var(--panel);
  backdrop-filter: blur(12px);
  border-radius: 20px;
  padding: 18px;
  margin-bottom: 22px;
  box-shadow: 0 8px 26px rgba(0,0,0,0.25);
  transition: all .3s ease;
}
.card-glass:hover { transform: translateY(-6px); box-shadow:0 12px 34px rgba(0,0,0,0.35); }

/* KPIs premium */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border-radius: 18px; color:white !important;
  box-shadow: 0 8px 22px rgba(0,0,0,0.25);
  transition: transform .3s ease;
}
[data-testid="stMetric"]:hover { transform:scale(1.03); }

/* Tab style */
.stTabs [role="tab"] { font-weight:800; color:var(--text)!important; padding:12px 22px; }
.stTabs [role="tab"][aria-selected="true"] {
  border-bottom: 3px solid var(--accent);
  background: rgba(255,255,255,0.08);
}

/* Plotly containers */
.js-plotly-plot .plot-container {
  border-radius:20px !important;
  box-shadow:0 12px 30px rgba(0,0,0,0.25) !important;
  padding:12px;
}

/* DataFrame tablas */
.stDataFrame [data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 2px solid var(--accent);
}

/* Footer corporativo */
.footer {
  text-align:center; padding:20px; margin-top:40px;
  font-size:14px; color:var(--text);
  border-top:1px solid rgba(255,255,255,0.15);
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# === Utils ===
def _convertir_img_base64(img):
    from io import BytesIO
    buf = BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception:
        buf.write(img)
    return base64.b64encode(buf.getvalue()).decode()



import os
import streamlit as st
from pathlib import Path
from PIL import Image
import base64
import requests
from io import BytesIO

def _convertir_img_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def _render_hero_con_logo():
    logo_img = None

    # 1) Intentar cargar desde carpeta del repo (ruta relativa)
    local_path = Path("images/Asecom/Asecom.png")
    if local_path.exists():
        try:
            logo_img = Image.open(local_path)
        except Exception as e:
            st.warning(f"No se pudo abrir el logo local: {e}")

    # 2) Si no existe local, intentar desde GitHub raw URL
    if logo_img is None:
        raw_url = ("https://raw.githubusercontent.com/"
                   "yyangs21/A3eC0Mc0mB3x_Yy/master/Asecom/Asecom.png")
        try:
            resp = requests.get(raw_url, timeout=10)
            if resp.status_code == 200:
                logo_img = Image.open(BytesIO(resp.content))
            else:
                st.warning("Logo no disponible desde URL raw (status code != 200)")
        except Exception as e:
            st.warning(f"Error al descargar logo desde GitHub: {e}")

    # 3) Convertir a base64 e insertar en HTML (si hay imagen)
    logo_html = ""
    if logo_img:
        try:
            logo_b64 = _convertir_img_base64(logo_img)
            logo_html = (
                f'<img src="data:image/png;base64,{logo_b64}" '
                f'style="height:100px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.3);" />'
            )
        except Exception as e:
            st.error(f"Error al convertir logo a base64: {e}")

    # 4) Renderizar el hero con o sin imagen
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(120deg,#0a1856 0%,#1b2a7c 50%,#0a1856 100%);
            padding: 20px 28px;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            color: white;
        ">
            <div>
                <h1 style="margin:0; font-size:40px;">📊 Dashboard ASECOM</h1>
                <p style="margin:0; font-size:18px;">Ventas 2023 – 2025 • Visualización Corporativa</p>
            </div>
            <div>{logo_html}</div>
        </div>
        """ ,
        unsafe_allow_html=True
    )

# Render hero al inicio
try:
    _render_hero_con_logo()
except Exception:
    pass

# ====== FIN INYECCIÓN PREMIUM ======

# ===================== MODO & ESTILOS PREMIUM =====================
# Toggle de tema (claro/oscuro)
with st.sidebar:
    st.markdown("### 🎨 Apariencia")
    modo_oscuro = st.toggle("🌙 Modo oscuro", value=False)

# CSS variables para ambos temas (incluye regla para hero en blanco)
light_css = """
<style>
/* ====== Sidebar en general ====== */
[data-testid="stSidebar"] { background: #0b1220; }
[data-testid="stSidebar"] * { color: #ffffff !important; }

/* ====== Inputs del sidebar ====== */
[data-testid="stSidebar"] .stMultiSelect > div,
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stTextInput > div,
[data-testid="stSidebar"] .stDateInput > div,
[data-testid="stSidebar"] .stSlider > div {
  background: #0f172a !important;
  border: 1px solid #334155 !important;
}
[data-baseweb="tag"] { background:#1f2937 !important; color:#fff !important; border-color:#334155 !important; }
[data-baseweb="select"] input, [data-baseweb="select"] div { color:#fff !important; }
[data-baseweb="menu"] { background:#0f172a !important; }
[data-baseweb="menu"] [role="option"] { color:#fff !important; }
[data-baseweb="menu"] [role="option"][aria-selected="true"] { background:#1f2937 !important; }
[data-testid="stCheckbox"] label p { color:#fff !important; }
[data-testid="stHelp"] { color:#e5e7eb !important; }

:root {
  --bg: #f6f8fb; --panel: #ffffff; --text: #008000; --muted:#4f5b66; --accent:#0a1856; --accent-2:#ffb347; --ring: rgba(255,127,17,0.25);
}
[data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }

/* Títulos globales en light */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

/* ✅ Fuerza texto blanco en el hero (gana por especificidad+!important) */
:root .stApp .asecom-hero, :root .stApp .asecom-hero * { color:#ffffff !important; }
</style>
"""

dark_css = """
<style>
:root { --bg:#0b1324; --panel:#111a2b; --text:#e6edf3; --muted:#9fb3c8; --accent:#0a1856; --accent-2:#ffb869; --ring: rgba(255,140,26,0.25); }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background:#050a14 !important; color: var(--text) !important; border-right:1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6, [data-testid="stSidebar"] label { color: var(--text) !important; }

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

/* ✅ Fuerza texto blanco en el hero */
:root .stApp .asecom-hero, :root .stApp .asecom-hero * { color:#ffffff !important; }
</style>
"""

st.markdown(dark_css if modo_oscuro else light_css, unsafe_allow_html=True)

# ===================== FIN ESTILOS =====================

# --- Tema para Plotly ---
def plotly_config_theme(fig, dark_mode=False):
    font_color = "#e6edf3" if dark_mode else "#90EE90"
    bg_color = "rgba(0,0,0,0)"
    fig.update_layout(
        template="simple_white",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title=dict(
            font=dict(size=26, family='Segoe UI', color=font_color),
            x=0.5, y=0.98, xanchor="center", yanchor="top"
        ),
        font=dict(color=font_color, family='Segoe UI', size=14),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)" if dark_mode else "rgba(255,255,255,0.6)",
            bordercolor="rgba(255,255,255,0.1)" if dark_mode else "rgba(0,0,0,0.1)", borderwidth=1
        ),
        margin=dict(l=100, r=80, t=160, b=90),
        hoverlabel=dict(font_color=font_color, bgcolor="#1e293b" if dark_mode else "#f7fbff", bordercolor="#ff7f11")
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)', zeroline=False,
                     title_font=dict(color=font_color), tickfont=dict(color=font_color))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)', zeroline=False,
                     title_font=dict(color=font_color), tickfont=dict(color=font_color))
    if fig.data:
        for tr in fig.data:
            if tr.type == 'bar':
                tr.update(marker_line_width=0, opacity=0.95)
            if tr.type in ('scatter', 'scattergl'):
                tr.update(mode='lines+markers', marker=dict(size=7), line=dict(width=2, shape='spline', smoothing=0.6))
            if tr.type == 'pie':
                labels = getattr(tr, 'labels', [])
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                tr.update(textposition='inside', textinfo='percent+label', hole=0.36, pull=[0.02]*len(labels))

ruta_logos = r"C:\\Users\\yyang\\Downloads"

def cargar_logo(nombre_archivo):
    ruta = os.path.join(ruta_logos, nombre_archivo)
    try:
        return Image.open(ruta)
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen {nombre_archivo}: {e}")
        return None

st.markdown("# 📊 Data 2023-ACTUALIDAD")

# --- Función para cargar datos ---
@st.cache_data(ttl=600)
def cargar_datos(hojas_seleccionadas=None):
    excel_path = '2025.xlsx'
    hojas_disponibles = ['MOVS 2023', 'MOVS 2024', 'MOVS 2025']
    if not hojas_seleccionadas:
        hojas = hojas_disponibles
    else:
        hojas = [h for h in hojas_disponibles if h in hojas_seleccionadas]

    lista_df = []
    for hoja in hojas:
        df = pd.read_excel(excel_path, sheet_name=hoja)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        df['Hoja'] = hoja
        lista_df.append(df)
    df_concat = pd.concat(lista_df, ignore_index=True)

    # Clasificación productos
    categoria_por_producto = {
        "🥤 Bebidas": [
            "ESPRESSO", "ESSPRESO", "EXPRESSO", "ICE COFFEE", "ICE LATTE", "LATTE", "CAFÉ", "CAFE", "CAPUCHINO", "CHAI",
            "FRAPPE", "CIMARRONA", "CHIMARRONA", "BOTELLA AGUA", "COCA COLA", "CUBETAZO", "FANTA", "SPRITE", "JUGO",
            "LICUADO", "HORCHATA", "ROSA DE JAMAICA", "SMOOTHIE", "HATSU GRANDE", "TE DE INFUSION", "CERVEZA",
            "CINZANO", "COPA", "VINO", "FLIGTH GREEN", "CODIGO INCORRECTO","AGUA PURA","AMERICANO","GASEOSAS","FRAPE","FRAPPÉ","FRESCO","LECHE",
            "SABORIZANTES","PEPSI","GASEOSA","PIÑA","BUBBLE GO","PREPARADO","COCACOLA","FRESCA","LIMONADA","MIX"
        ],
        "🍽️ Comidas": [
            "TOSTADAS A LA FRANCESA", "ALMUERZO", "CENA", "DESAYUNO", "CHOW MEIN", "PAN", "PAN CON", "PANINI", "PANQUEQUE",
            "TACOS", "SALCHICHA", "PACHES", "ENSALADA", "POLLO", "HUEVO", "OMELETTE", "LUNCH", "LASAGNA", "RAVIOLI",
            "CALZONE", "CHUCHITO", "DOBLADAS", "REFACCION", "REFACCIÓN", "FRUTA", "VEGETALES", "PORCION DE","CROISSANT","PORCION","CEVICHE","CAMARONES",
            "CHOCO","TOSTADOS","TRENZAS","CHILE","MONTADITO"
        ],
        "🍕 Pizzas": [
            "PIZZA", "PLAN DE VUELO", "ESCALA PERFECTA", "TICKET PERSONAL", "TRAYECTO COMPARTIDO", "VIAJE REDONDO",
            "TRAYECTO ESPECIAL", "VIAJE TOSCANO", "CLASE GO", "PALITROQUES", "CARNE EXTRA", "QUESO EXTRA",
            "DOBLE DESTINO","PORCION DE PIZZA","ESCALA","JET"
        ],
        "🍨 Postres": [
            "SELLO DULCE", "CHECK IN ROLL", "STRUDEL", "DONAS", "DONAS VARIEDAD", "PASTEL", "POLVOROSAS", "ROLES", "CUP CAKE",
            "CHESSCAKES", "PASTEL FRIO", "OFERTA GALLETA", "SELLO DE FRESA", "PASTEL DE NATA", "ROLES SABORIZADOS","ENCANELADOS","GRANIZADA"
        ],
        "🍪 Snacks y dulces": [
            "SNACK", "PAPALINA", "PLATANINA", "YUCA BOLSA", "LAYS", "CHUCHITO", "TRIDENT", "GALLETAS", "CORN FLAKES",
            "AVENA", "CHOCOLATE", "DOBLADAS", "TAMALITO", "TOSTADAS GENERAL","NATURAL MIX DE LA FINCA","DONA"
        ],
        "💼 Promociones": ["PROMO", "COMBO", "OFERTA", "DESCUENTO","+"],
        "🚗 Servicios / Otros": [
            "MANTENIMIENTO", "ENVÍO", "COMISIÓN", "QUINTAL DE CARTÓN", "QUINTAL DE NAYLON", "TARIMAS", "LAPTOP",
            "FOTOCOPIAS", "ALQUILER SALÓN", "SERVICIO DE TRASLADO","CHATARRA","EMPAQUE DE ALIMENTOS","LEÑA","RECARGA DE"
        ],
        "Canchas": [
            "ALQUILER DE CANCHA OUTSOURCING", "ALQUILER DE CANCHA HORA", "ALQUILER DE BALON", "AGUA PURA BOTELLA",
            "COCA COLA VARIEDAD LATA", "TORNEO 2025", "HORA Y MEDIA DE ALQUILER CANCHA 10-28", "KALORUB 30 G", "DUO DOLOKALORUB Y FRIORUB",
            "MEDIAS PARA JUGADOR", "TORNEO APERTURA 2025", "FARDO DE BOLSAS DE AGUA PURA", "MENSUALIDAD ESCUELA DE FUTBOL", "ESPINILLERA PARA ADULTO",
            "TARJETA AMARILLA", "TARJETA ROJA", "COFAL FUERTE 60GR", "KALORUB 60 MG", "Servicio de Arbitraje, hidratación, trofeos y medallas para campeonato",
            "Servicio de Arbitraje, hidratación, trofeos y medallas para campeonato","TORNEO CANCHA 10-28","PELOTA DE FUTBOL","Hora extra de entreno por campeonato",
            "SERVICIO POR ARBITRAJE","BOLSITAS ALGODÓN SUPERIOR","POPOROPOS PORCIÓN","UNIFORME DEPORTIVO PARA NIÑO","ANTICIPO DE CAMPEONATO DE FUTBOL",""
        ]
    }

    def clasificar_producto(descripcion):
        desc_upper = str(descripcion).upper()
        for categoria, claves in categoria_por_producto.items():
            for clave in claves:
                if clave in desc_upper:
                    return categoria
        return "❓ Sin categoría"

    df_concat['Categoría'] = df_concat['Descripción'].apply(clasificar_producto)

    # Cálculo IVA
    df_concat['IVA'] = (pd.to_numeric(df_concat['Valor'], errors='coerce').fillna(0) *
                        pd.to_numeric(df_concat['Cantidad'], errors='coerce').fillna(0)) - \
                        pd.to_numeric(df_concat['Total'], errors='coerce').fillna(0)

    # Limpieza
    df_concat['Tipo Movto.'] = df_concat['Tipo Movto.'].astype(str).str.strip()
    df_concat['UNIDAD'] = df_concat['UNIDAD'].astype(str).str.strip()
    df_concat['Total'] = pd.to_numeric(df_concat['Total'], errors='coerce').fillna(0)
    df_concat['Cantidad'] = pd.to_numeric(df_concat['Cantidad'], errors='coerce').fillna(0)
    return df_concat

# --- Sidebar filtros ---
st.sidebar.header("🔎 Filtros (opcionales)")
with st.sidebar.expander('Filtros y opciones', expanded=True):
    hojas_disponibles = ['MOVS 2023', 'MOVS 2024', 'MOVS 2025']
    hojas_seleccion = st.multiselect("Selecciona año(s) a analizar (hojas Excel)", options=hojas_disponibles, default=[])

df = cargar_datos(hojas_seleccion)

# === Filtro especial: Unificación exacta COMBEX-IM (una sola vez) ===
if 'Nombre' in df.columns:
    df['Nombre'] = df['Nombre'].astype(str).str.strip().replace({
        'ASOCIACION PARA EL DESARROLLO ECONOMICO Y SOCIAL DE AEROPUERTOS Y PUERTOS COMBEX-IM--': 'COMBEX-IM',
        '-COMBEX-IM--': 'COMBEX-IM',
        '242- ASOCIACION PARA EL DESARROLLO ECONOMICO Y SOCIAL DE AEROPUERTOS Y PUERTOS COMBEX-IM--': 'COMBEX-IM',
        'ASOCIACION PARA EL DESARROLLO ECONOMICO Y SOCIAL DE AEROPUERTOS Y PUERTOS COMBEX-IM': 'COMBEX-IM',
        '242- COMBEX-IM--': 'COMBEX-IM',
        'COMBEX-IM--': 'COMBEX-IM',
        '- COMBEX-IM--': 'COMBEX-IM',
        '- ASOCIACION PARA EL DESARROLLO ECONOMICO Y SOCIAL DE AEROPUERTOS Y PUERTOS COMBEX-IM--': 'COMBEX-IM'
    }, regex=False)

with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Opciones especiales")
    incluir_combex = st.checkbox("Incluir COMBEX-IM", value=True, help="Si lo desmarcas, se excluyen todos los registros de COMBEX-IM")

if not incluir_combex and 'Nombre' in df.columns:
    df = df[df['Nombre'] != 'COMBEX-IM']
# === Fin filtro especial ===

fecha_min, fecha_max = df['Fecha'].min(), df['Fecha'].max()
rango_fecha = st.sidebar.date_input("📅 Rango de Fechas", [fecha_min, fecha_max], format="YYYY-MM-DD")


def filtro_multiselect(columna, label):
    opciones = df[columna].dropna().unique()
    return st.sidebar.multiselect(f"🔹 {label}", options=sorted(opciones))

filtros = {
    "Categoría": filtro_multiselect("Categoría", "Categoría"),
    "Categoria PRD": filtro_multiselect("Categoria PRD", "Categoria PRD"),
    "Tipo Movto.": filtro_multiselect("Tipo Movto.", "Tipo Movto."),
    "No. Docto.": filtro_multiselect("No. Docto.", "No. Docto."),
    "Cod. Artículo": filtro_multiselect("Cod. Artículo", "Cod. Artículo"),
    "Descripción": filtro_multiselect("Descripción", "Descripción"),
    "UNIDAD": filtro_multiselect("UNIDAD", "UNIDAD"),
}

# Chips resumen de filtros activos (visual)
chips = []
if isinstance(rango_fecha, (list, tuple)) and len(rango_fecha) == 2:
    chips.append(f"Fechas: {pd.to_datetime(rango_fecha[0]).date()} → {pd.to_datetime(rango_fecha[1]).date()}")
for col, vals in filtros.items():
    if vals:
        val_short = ", ".join([str(v) for v in vals[:2]]) + (f" (+{len(vals)-2})" if len(vals) > 2 else "")
        chips.append(f"{col}: {val_short}")
if chips:
    st.markdown('<div class="card-glass"><h4>Resumen de filtros</h4><div class="chips">' +
                "".join([f"<span class='chip'>{c}</span>" for c in chips]) +
                '</div></div>', unsafe_allow_html=True)

# Aplicar filtros
if isinstance(rango_fecha, (list, tuple)) and len(rango_fecha) == 2:
    df_filtrado = df[(df['Fecha'] >= pd.to_datetime(rango_fecha[0])) & (df['Fecha'] <= pd.to_datetime(rango_fecha[1]))]
else:
    df_filtrado = df.copy()

for columna, valores in filtros.items():
    if valores:
        df_filtrado = df_filtrado[df_filtrado[columna].isin(valores)]

# Función calcular total neto

def calcular_total_vendido_neto(df_):
    ventas = df_[df_['Tipo Movto.'].isin([
        'VT-VENTA', 'CHEQUE', 'DEPOSITO', 'TRANSFERENCIAS', 'POS/VISALINK', '#N/A', 'EFECTIVO', 'CREDITO ASOCIADOS'
    ])]['Total'].sum()
    devoluciones = df_[df_['Tipo Movto.'] == 'RV-DEVOLUCION DE VENTAS']['Total'].sum()
    compras_credito = df_[df_['Tipo Movto.'].isin([
        'CP-COMPRAS AL CREDITO (PROVEEDORES)', 'CA-COMPRAS EN EFECTIVO (PROVEEDORES)'
    ])]['Total'].sum()
    ajustes_positivos = df_[df_['Tipo Movto.'] == 'PI-AJUSTE POSITIVO DE INVENTARIO']['Total'].sum()
    ajustes_negativos = df_[df_['Tipo Movto.'] == 'NI-AJUSTE NEGATIVO DE INVENTARIO']['Total'].sum()
    total_neto = ventas - devoluciones - compras_credito + ajustes_positivos - ajustes_negativos
    return total_neto

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["KPIs & Gráficos", "Comparación Comercial", "Predicción ML"])

with tab1:
    st.header("📈 KPIs y Gráficos")
    total_neto = calcular_total_vendido_neto(df_filtrado)
    iva_total = df_filtrado['IVA'].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Vendido Neto", f"Q{total_neto:,.2f}")
    col2.metric("🧾 Total IVA", f"Q{iva_total:,.2f}")

    kpi = st.selectbox("📌 Selecciona un KPI para visualizar", [
        "Total Vendido por Unidad y Tipo Movimiento",
        "Productos más vendidos (general)",
        "Productos más vendidos por unidad",
        "Clientes más frecuentes"
    ])

    if kpi == "Total Vendido por Unidad y Tipo Movimiento":
        tipos_mvto = [
            "VT-VENTA",
            "RV-DEVOLUCION DE VENTAS",
            "CP-COMPRAS AL CREDITO (PROVEEDORES)",
            "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
            "PI-AJUSTE POSITIVO DE INVENTARIO",
            "NI-AJUSTE NEGATIVO DE INVENTARIO",
            "CHEQUE", "DEPOSITO", "TRANSFERENCIAS", "POS/VISALINK", "#N/A", "EFECTIVO", "CREDITO ASOCIADOS"
        ]
        df_kpi = df_filtrado[df_filtrado["Tipo Movto."].isin(tipos_mvto)].copy()
        fig = px.histogram(
            df_kpi, x="UNIDAD", y="Total", histfunc="sum", color="Tipo Movto.",
            title="Total por Unidad y Tipo de Movimiento", barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2
        )
        plotly_config_theme(fig, dark_mode=modo_oscuro)
        st.plotly_chart(fig, use_container_width=True)

    elif kpi == "Productos más vendidos (general)":
        top_productos = df_filtrado.groupby("Descripción")["Cantidad"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_productos, x=top_productos.values, y=top_productos.index, orientation='h',
                     title="📦 Productos Más Vendidos (General)", labels={"x": "Cantidad"},
                     color_discrete_sequence=['#0b3d91'])
        plotly_config_theme(fig, dark_mode=modo_oscuro)
        st.plotly_chart(fig, use_container_width=True)

    elif kpi == "Productos más vendidos por unidad":
        unidades_filtradas = filtros["UNIDAD"]
        if unidades_filtradas:
            for unidad in unidades_filtradas:
                st.subheader(f"🎯 Unidad: {unidad}")
                df_unidad = df_filtrado[df_filtrado["UNIDAD"] == unidad]
                top_descripciones = (
                    df_unidad.groupby("Descripción")["Cantidad"].sum().sort_values(ascending=False).head(5).reset_index()
                )
                fig = px.pie(top_descripciones, names="Descripción", values="Cantidad",
                              title=f"🥧 Top Productos más vendidos en {unidad}", color_discrete_sequence=px.colors.sequential.Blues)
                plotly_config_theme(fig, dark_mode=modo_oscuro)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Por favor selecciona al menos una UNIDAD para ver este gráfico.")

    elif kpi == "Clientes más frecuentes":
        ventas_df = df_filtrado[df_filtrado["Tipo Movto."].isin([
            "VT-VENTA", "CHEQUE", "DEPOSITO", "TRANSFERENCIAS", "POS/VISALINK", "#N/A", "EFECTIVO", "CREDITO ASOCIADOS"
        ])]
        top_clientes = ventas_df.groupby('Nombre')['Total'].sum().reset_index()
        top_clientes = top_clientes.sort_values(by='Total', ascending=False).head(10)
        total_ventas = top_clientes['Total'].sum()
        total_general = calcular_total_vendido_neto(df_filtrado)
        otros_total = max(total_general - total_ventas, 0)
        grafica_data = top_clientes.copy()
        if otros_total > 0:
            otros_df = pd.DataFrame([{'Nombre': 'Otros Clientes', 'Total': otros_total}])
            grafica_data = pd.concat([grafica_data, otros_df], ignore_index=True)

        clientes_seleccionados = st.multiselect("🎯 Selecciona compradores a mostrar (Top 10)", options=top_clientes["Nombre"], default=None)
        clientes = grafica_data['Nombre'].tolist()
        totales = grafica_data['Total'].tolist()
        if clientes_seleccionados:
            seleccion = set(clientes_seleccionados)
            valores_ajustados = [total if nombre in seleccion else 0 for nombre, total in zip(clientes, totales)]
        else:
            seleccion = set(clientes)
            valores_ajustados = totales
        colors = ['#0b3d91' if nombre in seleccion else 'lightgrey' for nombre in clientes]
        plot_nombres, plot_valores, plot_colors = [], [], []
        for n, v, c in zip(clientes, valores_ajustados, colors):
            if v > 0:
                plot_nombres.append(n)
                plot_valores.append(v)
                plot_colors.append(c)
        fig_pie = px.pie(names=plot_nombres, values=plot_valores, title='Distribución del Total Vendido (Top 10 Clientes + Otros)', hole=0.3, color_discrete_sequence=plot_colors)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        plotly_config_theme(fig_pie, dark_mode=modo_oscuro)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Tabla
    st.markdown("### 📄 Registros Filtrados")
    st.dataframe(df_filtrado.reset_index(drop=True), use_container_width=True)

with tab2:
    st.header("📊 Comparación Comercial")
    st.markdown("Seleccione dos períodos para comparar las ventas netas por unidad y productos.")
    colA, colB = st.columns(2)
    with colA:
        mes1 = st.selectbox("Mes 1", options=list(range(1, 13)), index=datetime.now().month - 1)
        año1 = st.selectbox("Año 1", options=[2023, 2024, 2025], index=2)
    with colB:
        mes2 = st.selectbox("Mes 2", options=list(range(1, 13)), index=max(datetime.now().month - 2, 0))
        año2 = st.selectbox("Año 2", options=[2023, 2024, 2025], index=2)

    def ventas_mes_año(df, mes, año):
        df_temp = df[(df['Fecha'].dt.month == mes) & (df['Fecha'].dt.year == año)]
        tipos_mvto = [
            "VT-VENTA", "RV-DEVOLUCION DE VENTAS", "CP-COMPRAS AL CREDITO (PROVEEDORES)", "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
            "PI-AJUSTE POSITIVO DE INVENTARIO", "NI-AJUSTE NEGATIVO DE INVENTARIO", "CHEQUE", "DEPOSITO", "TRANSFERENCIAS",
            "POS/VISALINK", "#N/A", "EFECTIVO", "CREDITO ASOCIADOS"
        ]
        df_temp = df_temp[df_temp["Tipo Movto."].isin(tipos_mvto)]
        ventas = df_temp[df_temp['Tipo Movto.'].isin(['VT-VENTA', 'CHEQUE', 'DEPOSITO', 'TRANSFERENCIAS', 'POS/VISALINK', '#N/A', 'EFECTIVO', 'CREDITO ASOCIADOS'])].groupby('UNIDAD')['Total'].sum()
        devoluciones = df_temp[df_temp['Tipo Movto.'] == 'RV-DEVOLUCION DE VENTAS'].groupby('UNIDAD')['Total'].sum()
        compras_credito = df_temp[df_temp['Tipo Movto.'].isin(['CP-COMPRAS AL CREDITO (PROVEEDORES)','CA-COMPRAS EN EFECTIVO (PROVEEDORES)'])].groupby('UNIDAD')['Total'].sum()
        ajustes_pos = df_temp[df_temp['Tipo Movto.'] == 'PI-AJUSTE POSITIVO DE INVENTARIO'].groupby('UNIDAD')['Total'].sum()
        ajustes_neg = df_temp[df_temp['Tipo Movto.'] == 'NI-AJUSTE NEGATIVO DE INVENTARIO'].groupby('UNIDAD')['Total'].sum()
        total_neto = ventas.subtract(devoluciones, fill_value=0).subtract(compras_credito, fill_value=0).add(ajustes_pos, fill_value=0).subtract(ajustes_neg, fill_value=0)
        return total_neto.fillna(0)

    def ventas_productos_mes_año(df, mes, año):
        df_temp = df[(df['Fecha'].dt.month == mes) & (df['Fecha'].dt.year == año)]
        tipos_mvto = [
            "VT-VENTA", "RV-DEVOLUCION DE VENTAS", "CP-COMPRAS AL CREDITO (PROVEEDORES)", "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
            "PI-AJUSTE POSITIVO DE INVENTARIO", "NI-AJUSTE NEGATIVO DE INVENTARIO", "CHEQUE", "DEPOSITO", "TRANSFERENCIAS",
            "POS/VISALINK", "#N/A", "EFECTIVO", "CREDITO ASOCIADOS"
        ]
        df_temp = df_temp[df_temp["Tipo Movto."].isin(tipos_mvto)]
        df_temp['signo'] = df_temp['Tipo Movto.'].map({
            'VT-VENTA': 1, 'RV-DEVOLUCION DE VENTAS': -1, 'CP-COMPRAS AL CREDITO (PROVEEDORES)': -1, 'CA-COMPRAS EN EFECTIVO (PROVEEDORES)': -1,
            'PI-AJUSTE POSITIVO DE INVENTARIO': 1, 'NI-AJUSTE NEGATIVO DE INVENTARIO': -1, 'CHEQUE': 1, 'DEPOSITO': 1, 'TRANSFERENCIAS': 1,
            'POS/VISALINK': 1, '#N/A': 1, 'EFECTIVO': 1, 'CREDITO ASOCIADOS': 1,
        })
        df_temp['Total_Neto'] = df_temp['Total'] * df_temp['signo']
        ventas_prod = df_temp.groupby('Descripción')['Total_Neto'].sum()
        return ventas_prod.fillna(0)

    v1 = ventas_mes_año(df_filtrado, mes1, año1)
    v2 = ventas_mes_año(df_filtrado, mes2, año2)
    comparacion_df = pd.DataFrame({'Periodo 1': v1, 'Periodo 2': v2}).fillna(0)
    comparacion_df['Diferencia'] = comparacion_df['Periodo 2'] - comparacion_df['Periodo 1']
    comparacion_df = comparacion_df.reset_index()

    st.markdown(f"**Comparación de ventas netas por unidad:** {mes1}/{año1} vs {mes2}/{año2}")
    fig_comp = px.bar(comparacion_df, x='UNIDAD', y=['Periodo 1', 'Periodo 2'], barmode='group',
                      title="Ventas Netas por Unidad en Periodos Seleccionados",
                      labels={'value': 'Q Total Neto', 'UNIDAD': 'Unidad'}, color_discrete_sequence=px.colors.qualitative.Set2)
    plotly_config_theme(fig_comp, dark_mode=modo_oscuro)
    st.plotly_chart(fig_comp, use_container_width=True)

    vp1 = ventas_productos_mes_año(df_filtrado, mes1, año1)
    vp2 = ventas_productos_mes_año(df_filtrado, mes2, año2)
    comp_prod_df = pd.DataFrame({'Periodo 1': vp1, 'Periodo 2': vp2}).fillna(0)
    comp_prod_df['Diferencia'] = comp_prod_df['Periodo 2'] - comp_prod_df['Periodo 1']
    comp_prod_df = comp_prod_df.sort_values('Periodo 2', ascending=False).reset_index()

    st.markdown(f"**Comparación de productos vendidos:** {mes1}/{año1} vs {mes2}/{año2}")
    top_prod_df = comp_prod_df.head(15)
    fig_prod = px.bar(top_prod_df, y='Descripción', x=['Periodo 1', 'Periodo 2'], orientation='h', barmode='group', text_auto=True, height=900,
                      title="Top 15 Productos Más Vendidos en los Periodos Seleccionados",
                      labels={'value': 'Q Total Neto', 'Descripción': 'Producto'}, color_discrete_sequence=px.colors.qualitative.Bold)
    fig_prod.update_layout(yaxis={'categoryorder':'total ascending'}, font=dict(size=14))
    plotly_config_theme(fig_prod, dark_mode=modo_oscuro)
    st.plotly_chart(fig_prod, use_container_width=True)

    st.dataframe(comparacion_df.style.format({"Periodo 1": "Q{:,.2f}", "Periodo 2": "Q{:,.2f}", "Diferencia": "Q{:,.2f}"}), use_container_width=True)

with tab3:
    st.header("🤖 Predicción y Modelos ML")
    modelo_sel = st.selectbox("Selecciona el modelo de predicción", options=["Prophet", "XGBoost"])

    unidades_ml = df['UNIDAD'].unique().tolist()
    unidad_ml = st.selectbox("Selecciona la UNIDAD para predicción", options=unidades_ml,
                             index=unidades_ml.index('Pizza Go') if 'Pizza Go' in unidades_ml else 0)

    if str(unidad_ml).lower() == 'pizza go':
        df_ml = df[(df['UNIDAD'] == unidad_ml) & (df['Fecha'].dt.year == 2025)].copy()
    else:
        df_ml = df[df['UNIDAD'] == unidad_ml].copy()

    df_ml['Tipo Movto.'] = df_ml['Tipo Movto.'].astype(str).str.strip()
    tipos_mvto = [
        "VT-VENTA","RV-DEVOLUCION DE VENTAS","CP-COMPRAS AL CREDITO (PROVEEDORES)",
        "CA-COMPRAS EN EFECTIVO (PROVEEDORES)","PI-AJUSTE POSITIVO DE INVENTARIO",
        "NI-AJUSTE NEGATIVO DE INVENTARIO","CHEQUE","DEPOSITO","TRANSFERENCIAS",
        "POS/VISALINK","#N/A","EFECTIVO","CREDITO ASOCIADOS"
    ]
    df_ml = df_ml[df_ml['Tipo Movto.'].isin(tipos_mvto)]

    df_ml['signo'] = df_ml['Tipo Movto.'].map({
        'VT-VENTA': 1,'RV-DEVOLUCION DE VENTAS': -1,'CP-COMPRAS AL CREDITO (PROVEEDORES)': -1,
        'CA-COMPRAS EN EFECTIVO (PROVEEDORES)': -1,'PI-AJUSTE POSITIVO DE INVENTARIO': 1, 'NI-AJUSTE NEGATIVO DE INVENTARIO': -1,
        'CHEQUE': 1,'DEPOSITO': 1,'TRANSFERENCIAS': 1, 'POS/VISALINK': 1,'#N/A': 1,'EFECTIVO': 1,'CREDITO ASOCIADOS': 1,
    })
    df_ml['Total_Neto'] = df_ml['Total'] * df_ml['signo']

    st.markdown('<p style="font-weight:700; font-size:16px;">Selecciona tipo de predicción:</p>', unsafe_allow_html=True)
    tipo_pred = option_menu(
        menu_title=None, options=["Diaria", "Mensual"], icons=["calendar-day", "calendar"],
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "0px", "background-color": "#f0f2f6"},
                "nav-link": {"font-size": "16px", "color": "black", "background-color": "#f0f2f6", "margin":"0px"},
                "nav-link-selected": {"color": "white", "background-color": "#003049"}}
    )

    if tipo_pred == "Diaria":
        df_agg = df_ml.groupby('Fecha')['Total_Neto'].sum().reset_index().rename(columns={'Fecha':'ds','Total_Neto':'y'})
        df_agg = df_agg.sort_values('ds')
    else:
        df_ml['AñoMes'] = df_ml['Fecha'].dt.to_period('M')
        df_agg = df_ml.groupby('AñoMes')['Total_Neto'].sum().reset_index()
        df_agg['ds'] = df_agg['AñoMes'].dt.to_timestamp()
        df_agg = df_agg.rename(columns={'Total_Neto':'y'}).sort_values('ds')

    fecha_minima, fecha_maxima = df_ml['Fecha'].min(), df_ml['Fecha'].max()
    fecha_inicio_pred = st.date_input("Fecha desde la que deseas predecir (inclusive):", value=fecha_maxima, min_value=fecha_minima, max_value=fecha_maxima)
    horizonte = st.slider("Horizonte de predicción (días o meses según tipo)", min_value=1, max_value=60 if tipo_pred=="Diaria" else 12, value=30 if tipo_pred=="Diaria" else 3)

    df_train = df_agg[df_agg['ds'] <= pd.to_datetime(fecha_inicio_pred)].copy()
    df_test_start = df_train['ds'].max()
    fechas_futuras = pd.date_range(start=df_test_start + (pd.Timedelta(days=1) if tipo_pred=="Diaria" else pd.offsets.MonthBegin(1)),
                                   periods=horizonte, freq='D' if tipo_pred=="Diaria" else 'MS')

    if st.button("Generar predicción"):
        if modelo_sel == "Prophet":
            with st.spinner("Entrenando modelo Prophet..."):
                model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
                model.fit(df_train)
                futuro = model.make_future_dataframe(periods=horizonte, freq='D' if tipo_pred=="Diaria" else 'MS')
                forecast = model.predict(futuro)
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(title=f"Predicción de ventas netas para {unidad_ml} con Prophet", font_color='red')
                plotly_config_theme(fig1, dark_mode=modo_oscuro)
                st.plotly_chart(fig1, use_container_width=True)
        elif modelo_sel == "XGBoost":
            with st.spinner("Entrenando modelo XGBoost..."):
                df_train_feat = df_train.copy()
                df_train_feat['month'] = df_train_feat['ds'].dt.month
                df_train_feat['year'] = df_train_feat['ds'].dt.year
                if tipo_pred == "Diaria":
                    df_train_feat['dayofweek'] = df_train_feat['ds'].dt.dayofweek
                    df_train_feat['dayofmonth'] = df_train_feat['ds'].dt.day
                    features = ['dayofweek','dayofmonth','month','year']
                else:
                    features = ['month','year']
                X_train, y_train = df_train_feat[features], df_train_feat['y']

                # Validación simple
                cutoff = df_train_feat['ds'].max() - (pd.Timedelta(days=30) if tipo_pred=="Diaria" else pd.offsets.MonthBegin(2))
                train_val = df_train_feat[df_train_feat['ds'] <= cutoff]
                valid_val = df_train_feat[df_train_feat['ds'] > cutoff]

                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
                model.fit(train_val[features], train_val['y'])

                if not valid_val.empty:
                    y_val_pred = model.predict(valid_val[features])
                    rmse_val = np.sqrt(mean_squared_error(valid_val['y'], y_val_pred))
                    st.write(f"RMSE validación: {rmse_val:.2f}")

                # Futuro
                df_future = pd.DataFrame({'ds': fechas_futuras})
                df_future['month'] = df_future['ds'].dt.month
                df_future['year'] = df_future['ds'].dt.year
                if tipo_pred=="Diaria":
                    df_future['dayofweek'] = df_future['ds'].dt.dayofweek
                    df_future['dayofmonth'] = df_future['ds'].dt.day
                X_future = df_future[features]
                preds_future = model.predict(X_future)

                fig2 = px.line(title=f"Predicción XGBoost para {unidad_ml}")
                fig2.add_scatter(x=df_train_feat['ds'], y=df_train_feat['y'], mode='lines+markers', name='Histórico')
                fig2.add_scatter(x=fechas_futuras, y=preds_future, mode='lines+markers', name='Predicción')
                plotly_config_theme(fig2, dark_mode=modo_oscuro)
                st.plotly_chart(fig2, use_container_width=True)

# Footer
try:
    st.markdown("""
    <div class="footer">
    ASECOM Dashboard Premium © 2025 • Desarrollado con ❤ en Streamlit
    </div>
    """, unsafe_allow_html=True)
except Exception:
    pass



