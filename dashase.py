import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
from datetime import datetime
import numpy as np
from streamlit_option_menu import option_menu
import requests

# ML imports
from prophet import Prophet
from prophet.plot import plot_plotly
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Dashboard Ventas Interactivo", layout="wide")

# --- Estilos CSS personalizados (fondo blanco profesional) ---
st.markdown("""
<style>
/* --- Fondo general blanco y texto principal oscuro --- */
[data-testid="stAppViewContainer"] {
    background: #ffffff;
    color: #111111;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Texto de etiquetas y controles dentro del cuerpo principal */
.stSelectbox label, 
.stMultiSelect label,
.stSlider label, 
.stRadio label, 
.stNumberInput label,
.stDateInput label {
    color: #FF0000 !important;
    font-weight: 600;
}
/* Cambiar color del texto de los radios a negro */
div[data-baseweb="radio"] span {
    color: black !important;
    font-weight: bold;
    font-size: 16px;
}

/* Opcional: cambiar color del círculo seleccionado */
div[data-baseweb="radio"] input:checked + span {
    background-color: #FF0000 !important; /* círculo relleno al seleccionar */
}

/* Opcional: borde del círculo no seleccionado */
div[data-baseweb="radio"] input + span {
    border-color: #FF0000 !important;
}
/* Texto de placeholder y opciones en selects */
.stSelectbox div[data-baseweb="select"] * ,
.stMultiSelect div[data-baseweb="select"] * {
    color: #FF0000 !important;
}

/* --- Encabezados --- */
/* h2 (st.header) en negro */
h2 {
    color: #000000 !important;
    font-weight: 700;
}
/* Otros encabezados (h1, h3, h4, h5) en azul oscuro */
h1, h3, h4, h5 {
    color: #0b3d91;
    font-weight: 700;
}

/* --- Barra lateral --- */
[data-testid="stSidebar"] {
    background: #0a2a66;  /* azul oscuro intenso */
    color: #cce4f7;       /* texto azul claro para contraste */
    font-weight: 600;
    padding: 1rem 1rem 2rem 1rem;
}

/* Textos de labels y filtros en sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiselect label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #cce4f7 !important;
    font-weight: 600;
}

/* Inputs (select, multiselect, radio, checkbox) en sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiselect > div > div,
[data-testid="stSidebar"] .stRadio > div > div,
[data-testid="stSidebar"] .stCheckbox > div > div {
    background-color: #163e8a !important; /* azul medio */
    color: #cce4f7 !important;            /* texto azul claro */
    border-radius: 6px;
    padding: 4px 8px;
}

/* --- Métricas (KPIs) --- */
[data-testid="stMetricValue"] {
    color: #0b3d91;
    font-weight: 700;
}

/* --- Botones, selects y multiselect (en cuerpo principal) --- */
.stSelectbox, .stMultiselect, .stButton {
    border-radius: 8px;
    background-color: #cce4f7; /* azul claro */
    color: #0b3d91;
    font-weight: 600;
}

/* --- Tabla --- */
.stDataFrame div[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #0b3d91; /* azul oscuro */
    background-color: #f9fbfe;  /* azul muy claro */
    color: #111111;
}

/* --- Texto en gráficos Plotly --- */
svg text, .xtick, .ytick {
    fill: #111111 !important;
    font-weight: 700;
}

/* Títulos de gráficos */
.plotly .main-svg .g-gtitle {
    fill: #0b3d91 !important;
    font-weight: 700;
}

/* --- Scroll personalizado para tablas --- */
div[data-testid="stDataFrame"] > div > div {
    scrollbar-color: #0b3d91 #f9fbfe;
    scrollbar-width: thin;
}
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar {
    width: 8px;
}
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar-track {
    background: #f9fbfe;
}
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar-thumb {
    background-color: #0b3d91;
    border-radius: 10px;
    border: 2px solid #cce4f7;
}
            /* Tabs (KPIs & Gráficos, Comparación Comercial, Predicción ML) */
.stTabs [role="tab"] {
    color: #000000 !important;  /* Texto negro */
    font-weight: 700;
}

/* Tab seleccionada con borde azul oscuro */
.stTabs [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #0b3d91 !important;
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)




# --- Función para aplicar tema a gráficos Plotly ---
def plotly_config_theme(fig):
    # Aplicar configuración común
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0)',  # fondo transparente para fondo blanco
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='red', family='Segoe UI', size=14),  # texto rojo general
        title_font=dict(color='red', size=18, family='Segoe UI'),
        legend_font=dict(color='red'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,0,0,0.1)',
            zerolinecolor='rgba(255,0,0,0.2)',
            title_font_color='red',
            tickfont_color='red',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,0,0,0.1)',
            zerolinecolor='rgba(255,0,0,0.2)',
            title_font_color='red',
            tickfont_color='red',
        ),
        hoverlabel=dict(
            font_color='red',
            bgcolor='white',
            bordercolor='red'
        )
    )

    # Ajustes específicos según tipo de gráfico para update_traces
    if fig.data:
        for trace in fig.data:
            if trace.type == 'pie':
                # En pie charts ponemos texto dentro con porcentaje y etiqueta
                fig.update_traces(textposition='inside', textinfo='percent+label', selector=dict(type='pie'))
                fig.update_traces(hoverinfo='label+percent+name', selector=dict(type='pie'))
            elif trace.type == 'bar':
                # En barras mostramos texto arriba
                fig.update_traces(textposition='outside', textfont_color='red', selector=dict(type='bar'))
            elif trace.type == 'scatter':
                pass


# --- Ruta base donde están los logos ---


def cargar_logo_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen desde {url}: {e}")
        return None

# Mostrar logo
url_logo_default = "https://raw.githubusercontent.com/yyangs21/A3eC0Mc0mB3x_Yy/master/logos/Asecom.png"
logo_asecom = cargar_logo_url(url_logo_default)
if logo_asecom:
    st.image(logo_asecom, width=720, caption="UNIDADES ASECOM")



st.markdown("# 📊 Dashboard Interactivo de Ventas 2023-2025")

# --- Función para cargar datos ---
@st.cache_data(ttl=600)
def cargar_datos(hojas_seleccionadas=None):
    excel_path = '2025.xlsx'
    hojas_disponibles = ['MOVS 2023', 'MOVS 2024', 'MOVS 2025']
    if not hojas_seleccionadas or len(hojas_seleccionadas) == 0:
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

    # Clasificación productos (igual que original)
    categoria_por_producto = { "🥤 Bebidas": [
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
        "🧁 Postres": [
            "SELLO DULCE", "CHECK IN ROLL", "STRUDEL", "DONAS", "DONAS VARIEDAD", "PASTEL", "POLVOROSAS", "ROLES", "CUP CAKE",
            "CHESSCAKES", "PASTEL FRIO", "OFERTA GALLETA", "SELLO DE FRESA", "PASTEL DE NATA", "ROLES SABORIZADOS","ENCANELADOS","GRANIZADA"
        ],
        "🍪 Snacks y dulces": [
            "SNACK", "PAPALINA", "PLATANINA", "YUCA BOLSA", "LAYS", "CHUCHITO", "TRIDENT", "GALLETAS", "CORN FLAKES",
            "AVENA", "CHOCOLATE", "DOBLADAS", "TAMALITO", "TOSTADAS GENERAL","NATURAL MIX DE LA FINCA","DONA"
        ],
        "💼 Promociones": [
            "PROMO", "COMBO", "OFERTA", "DESCUENTO","+" 
        ],
        "🚗 Servicios / Otros": [
            "MANTENIMIENTO", "ENVÍO", "COMISIÓN", "QUINTAL DE CARTÓN", "QUINTAL DE NAYLON", "TARIMAS", "LAPTOP",
            "FOTOCOPIAS", "ALQUILER SALÓN", "SERVICIO DE TRASLADO","CHATARRA","EMPAQUE DE ALIMENTOS","LEÑA","RECARGA DE"
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
    df_concat['IVA'] = (df_concat['Valor'] * df_concat['Cantidad']) - df_concat['Total']

    # Limpieza espacios
    df_concat['Tipo Movto.'] = df_concat['Tipo Movto.'].astype(str).str.strip()
    df_concat['UNIDAD'] = df_concat['UNIDAD'].astype(str).str.strip()

    df_concat['Total'] = pd.to_numeric(df_concat['Total'], errors='coerce').fillna(0)
    df_concat['Cantidad'] = pd.to_numeric(df_concat['Cantidad'], errors='coerce').fillna(0)

    return df_concat

# --- Sidebar filtros ---
st.sidebar.header("🔎 Filtros (opcionales)")

hojas_disponibles = ['MOVS 2023', 'MOVS 2024', 'MOVS 2025']
hojas_seleccion = st.sidebar.multiselect("Selecciona año(s) a analizar (hojas Excel)", options=hojas_disponibles, default=[])

df = cargar_datos(hojas_seleccion)

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
# --- Logos dinámicos según UNIDAD filtrada ---
unidad_sel = None
if filtros.get("UNIDAD") and len(filtros["UNIDAD"]) == 1:
    unidad_sel = filtros["UNIDAD"][0].upper()
    # Mapear nombre de unidad a URL de logo en GitHub/Streamlit Cloud
    logos_unidades_url = {
        "GO CAFE": "https://raw.githubusercontent.com/tu_usuario/tu_repo/master/Cafe%20Go.png",
        "CAFETERIA": "https://raw.githubusercontent.com/tu_usuario/tu_repo/master/Cafeteria.png",
        "PIZZA GO": "https://raw.githubusercontent.com/tu_usuario/tu_repo/master/Pizza%20Go.png"
    }
    url_logo = logos_unidades_url.get(unidad_sel, "https://raw.githubusercontent.com/tu_usuario/tu_repo/master/Asecom.png")
else:
    url_logo = "https://raw.githubusercontent.com/tu_usuario/tu_repo/master/Asecom2.png"  # logo por defecto

# --- Cargar y mostrar logo filtrado ---
logo_img = cargar_logo_url(url_logo)
if logo_img:
    st.image(logo_img, width=350, caption=f"Logo {unidad_sel}" if unidad_sel else "UNIDADES ASECOM")


# Aplicar filtros
df_filtrado = df[
    (df['Fecha'] >= pd.to_datetime(rango_fecha[0])) &
    (df['Fecha'] <= pd.to_datetime(rango_fecha[1]))
]

for columna, valores in filtros.items():
    if valores:
        df_filtrado = df_filtrado[df_filtrado[columna].isin(valores)]

# Función calcular total neto
def calcular_total_vendido_neto(df_):
    ventas = df_[df_['Tipo Movto.'] == 'VT-VENTA']['Total'].sum()
    devoluciones = df_[df_['Tipo Movto.'] == 'RV-DEVOLUCION DE VENTAS']['Total'].sum()
    compras_credito = df_[
        df_['Tipo Movto.'].isin([
            'CP-COMPRAS AL CREDITO (PROVEEDORES)', 
            'CA-COMPRAS EN EFECTIVO (PROVEEDORES)'
        ])
    ]['Total'].sum()
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
            "NI-AJUSTE NEGATIVO DE INVENTARIO"
        ]
        df_kpi = df_filtrado[df_filtrado["Tipo Movto."].isin(tipos_mvto)].copy()

        fig = px.histogram(
            df_kpi,
            x="UNIDAD", y="Total", histfunc="sum", color="Tipo Movto.",
            title="Total por Unidad y Tipo de Movimiento",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        plotly_config_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    elif kpi == "Productos más vendidos (general)":
        top_productos = df_filtrado.groupby("Descripción")["Cantidad"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_productos, x=top_productos.values, y=top_productos.index, orientation='h',
                    title="📦 Productos Más Vendidos (General)", labels={"x": "Cantidad"},
                    color_discrete_sequence=['#0b3d91'])
        plotly_config_theme(fig)
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
                fig = px.pie(
                    top_descripciones,
                    names="Descripción",
                    values="Cantidad",
                    title=f"🥧 Top Productos más vendidos en {unidad}",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                plotly_config_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Por favor selecciona al menos una UNIDAD para ver este gráfico.")

    elif kpi == "Clientes más frecuentes":
        ventas_df = df_filtrado[df_filtrado["Tipo Movto."] == "VT-VENTA"]
        top_clientes = ventas_df.groupby('Nombre')['Total'].sum().reset_index()
        top_clientes = top_clientes.sort_values(by='Total', ascending=False).head(10)

        total_ventas = top_clientes['Total'].sum()
        total_general = calcular_total_vendido_neto(df_filtrado)
        otros_total = max(total_general - total_ventas, 0)

        grafica_data = top_clientes.copy()
        if otros_total > 0:
            otros_df = pd.DataFrame([{'Nombre': 'Otros Clientes', 'Total': otros_total}])
            grafica_data = pd.concat([grafica_data, otros_df], ignore_index=True)

        clientes_seleccionados = st.multiselect(
            "🎯 Selecciona compradores a mostrar (Top 10)",
            options=top_clientes["Nombre"],
            default=None
        )

        clientes = grafica_data['Nombre'].tolist()
        totales = grafica_data['Total'].tolist()

        if clientes_seleccionados:
            seleccion = set(clientes_seleccionados)
            valores_ajustados = [
                total if nombre in seleccion else 0
                for nombre, total in zip(clientes, totales)
            ]
        else:
            seleccion = set(clientes)
            valores_ajustados = totales

        colors = [
            '#0b3d91' if nombre in seleccion else 'lightgrey'
            for nombre in clientes
        ]

        plot_nombres = []
        plot_valores = []
        plot_colors = []

        for n, v, c in zip(clientes, valores_ajustados, colors):
            if v > 0:
                plot_nombres.append(n)
                plot_valores.append(v)
                plot_colors.append(c)

        fig_pie = px.pie(
            names=plot_nombres,
            values=plot_valores,
            title='Distribución del Total Vendido (Top 10 Clientes + Otros)',
            hole=0.3,
            color_discrete_sequence=plot_colors
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        plotly_config_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Mostrar tabla filtrada
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
        mes2 = st.selectbox("Mes 2", options=list(range(1, 13)), index=datetime.now().month - 2 if datetime.now().month > 1 else 0)
        año2 = st.selectbox("Año 2", options=[2023, 2024, 2025], index=2)

    # --- Función para ventas netas por unidad ---
    def ventas_mes_año(df, mes, año):
        df_temp = df[(df['Fecha'].dt.month == mes) & (df['Fecha'].dt.year == año)]
        tipos_mvto = [
            "VT-VENTA",
            "RV-DEVOLUCION DE VENTAS",
            "CP-COMPRAS AL CREDITO (PROVEEDORES)",
            "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
            "PI-AJUSTE POSITIVO DE INVENTARIO",
            "NI-AJUSTE NEGATIVO DE INVENTARIO"
        ]
        df_temp = df_temp[df_temp["Tipo Movto."].isin(tipos_mvto)]
        ventas = df_temp[df_temp['Tipo Movto.'] == 'VT-VENTA'].groupby('UNIDAD')['Total'].sum()
        devoluciones = df_temp[df_temp['Tipo Movto.'] == 'RV-DEVOLUCION DE VENTAS'].groupby('UNIDAD')['Total'].sum()
        compras_credito = df_temp[df_temp['Tipo Movto.'].isin(['CP-COMPRAS AL CREDITO (PROVEEDORES)','CA-COMPRAS EN EFECTIVO (PROVEEDORES)'])].groupby('UNIDAD')['Total'].sum()
        ajustes_pos = df_temp[df_temp['Tipo Movto.'] == 'PI-AJUSTE POSITIVO DE INVENTARIO'].groupby('UNIDAD')['Total'].sum()
        ajustes_neg = df_temp[df_temp['Tipo Movto.'] == 'NI-AJUSTE NEGATIVO DE INVENTARIO'].groupby('UNIDAD')['Total'].sum()

        total_neto = ventas.subtract(devoluciones, fill_value=0)\
            .subtract(compras_credito, fill_value=0)\
            .add(ajustes_pos, fill_value=0)\
            .subtract(ajustes_neg, fill_value=0)
        return total_neto.fillna(0)

    # --- Función para ventas netas por producto ---
    def ventas_productos_mes_año(df, mes, año):
        df_temp = df[(df['Fecha'].dt.month == mes) & (df['Fecha'].dt.year == año)]
        tipos_mvto = [
            "VT-VENTA",
            "RV-DEVOLUCION DE VENTAS",
            "CP-COMPRAS AL CREDITO (PROVEEDORES)",
            "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
            "PI-AJUSTE POSITIVO DE INVENTARIO",
            "NI-AJUSTE NEGATIVO DE INVENTARIO"
        ]
        df_temp = df_temp[df_temp["Tipo Movto."].isin(tipos_mvto)]
        df_temp['signo'] = df_temp['Tipo Movto.'].map({
            'VT-VENTA': 1,
            'RV-DEVOLUCION DE VENTAS': -1,
            'CP-COMPRAS AL CREDITO (PROVEEDORES)': -1,
            'CA-COMPRAS EN EFECTIVO (PROVEEDORES)': -1,
            'PI-AJUSTE POSITIVO DE INVENTARIO': 1,
            'NI-AJUSTE NEGATIVO DE INVENTARIO': -1,
        })
        df_temp['Total_Neto'] = df_temp['Total'] * df_temp['signo']
        ventas_prod = df_temp.groupby('Descripción')['Total_Neto'].sum()
        return ventas_prod.fillna(0)

    # --- Comparación por unidad ---
    v1 = ventas_mes_año(df_filtrado, mes1, año1)
    v2 = ventas_mes_año(df_filtrado, mes2, año2)
    comparacion_df = pd.DataFrame({'Periodo 1': v1, 'Periodo 2': v2}).fillna(0)
    comparacion_df['Diferencia'] = comparacion_df['Periodo 2'] - comparacion_df['Periodo 1']
    comparacion_df = comparacion_df.reset_index()

    st.markdown(f"**Comparación de ventas netas por unidad:** {mes1}/{año1} vs {mes2}/{año2}")
    fig_comp = px.bar(comparacion_df, x='UNIDAD', y=['Periodo 1', 'Periodo 2'], barmode='group',
                      title="Ventas Netas por Unidad en Periodos Seleccionados",
                      labels={'value': 'Q Total Neto', 'UNIDAD': 'Unidad'},
                      color_discrete_sequence=px.colors.qualitative.Set2)
    plotly_config_theme(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- Comparación por productos ---
    vp1 = ventas_productos_mes_año(df_filtrado, mes1, año1)
    vp2 = ventas_productos_mes_año(df_filtrado, mes2, año2)
    comp_prod_df = pd.DataFrame({'Periodo 1': vp1, 'Periodo 2': vp2}).fillna(0)
    comp_prod_df['Diferencia'] = comp_prod_df['Periodo 2'] - comp_prod_df['Periodo 1']
    comp_prod_df = comp_prod_df.sort_values('Periodo 2', ascending=False).reset_index()

    st.markdown(f"**Comparación de productos vendidos:** {mes1}/{año1} vs {mes2}/{año2}")
    top_prod_df = comp_prod_df.head(15)

    fig_prod = px.bar(
        top_prod_df,
        y='Descripción',
        x=['Periodo 1', 'Periodo 2'],
        orientation='h',
        barmode='group',
        text_auto=True,
        height=900,
        title="Top 15 Productos Más Vendidos en los Periodos Seleccionados",
        labels={'value': 'Q Total Neto', 'Descripción': 'Producto'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_prod.update_layout(yaxis={'categoryorder':'total ascending'}, font=dict(size=14))
    plotly_config_theme(fig_prod)
    st.plotly_chart(fig_prod, use_container_width=True)

    # --- Tabla resumen por unidad ---
    st.dataframe(comparacion_df.style.format({"Periodo 1": "Q{:,.2f}", "Periodo 2": "Q{:,.2f}", "Diferencia": "Q{:,.2f}"}), use_container_width=True)



   
with tab3:
    st.header("🤖 Predicción y Modelos ML")



    modelo_sel = st.selectbox("Selecciona el modelo de predicción", options=["Prophet", "XGBoost"])


    # Opciones para ML
    unidades_ml = df['UNIDAD'].unique().tolist()
    unidad_ml = st.selectbox("Selecciona la UNIDAD para predicción", options=unidades_ml, index=unidades_ml.index('Pizza Go') if 'Pizza Go' in unidades_ml else 0)

    # Restricción: si unidad es Pizza Go, usar solo datos 2025
    if unidad_ml.lower() == 'pizza go':
        df_ml = df[(df['UNIDAD'] == unidad_ml) & (df['Fecha'].dt.year == 2025)].copy()
    else:
        df_ml = df[df['UNIDAD'] == unidad_ml].copy()

    df_ml['Tipo Movto.'] = df_ml['Tipo Movto.'].astype(str).str.strip()
    tipos_mvto = [
        "VT-VENTA",
        "RV-DEVOLUCION DE VENTAS",
        "CP-COMPRAS AL CREDITO (PROVEEDORES)",
        "CA-COMPRAS EN EFECTIVO (PROVEEDORES)",
        "PI-AJUSTE POSITIVO DE INVENTARIO",
        "NI-AJUSTE NEGATIVO DE INVENTARIO"
    ]
    df_ml = df_ml[df_ml['Tipo Movto.'].isin(tipos_mvto)]

    # Calculamos Total Neto diario (VT - RV - CP/CA + PI - NI)
    df_ml['signo'] = df_ml['Tipo Movto.'].map({
        'VT-VENTA': 1,
        'RV-DEVOLUCION DE VENTAS': -1,
        'CP-COMPRAS AL CREDITO (PROVEEDORES)': -1,
        'CA-COMPRAS EN EFECTIVO (PROVEEDORES)': -1,
        'PI-AJUSTE POSITIVO DE INVENTARIO': 1,
        'NI-AJUSTE NEGATIVO DE INVENTARIO': -1,
    })
    df_ml['Total_Neto'] = df_ml['Total'] * df_ml['signo']

    # --- Selección tipo de predicción ---
    st.markdown('<p style="color:black; font-weight:bold; font-size:16px;">Selecciona tipo de predicción:</p>', unsafe_allow_html=True)
    tipo_pred = option_menu(
    menu_title=None,  # Sin título extra
    options=["Diaria", "Mensual"],
    icons=["calendar-day", "calendar"],  # íconos opcionales
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0px", "background-color": "#f0f2f6"},
        "nav-link": {"font-size": "16px", "color": "black", "background-color": "#f0f2f6", "margin":"0px"},
        "nav-link-selected": {"color": "white", "background-color": "#003049"},
    }
)

    # --- Fecha inicial para predicción ---
    fecha_minima = df_ml['Fecha'].min()
    fecha_maxima = df_ml['Fecha'].max()
    fecha_inicio_pred = st.date_input("Fecha desde la que deseas predecir (inclusive):", value=fecha_maxima, min_value=fecha_minima, max_value=fecha_maxima)

    horizonte = st.slider("Horizonte de predicción (días o meses según tipo)", min_value=1, max_value=60, value=30)

    # Preparar datos según tipo de predicción
    if tipo_pred == "Diaria":
        # Agrupar por día
        df_agg = df_ml.groupby('Fecha')['Total_Neto'].sum().reset_index()
        df_agg = df_agg.rename(columns={'Fecha':'ds', 'Total_Neto':'y'})
        df_agg = df_agg.sort_values('ds')
    else:
        # Agrupar por mes
        df_ml['AñoMes'] = df_ml['Fecha'].dt.to_period('M')
        df_agg = df_ml.groupby('AñoMes')['Total_Neto'].sum().reset_index()
        df_agg['ds'] = df_agg['AñoMes'].dt.to_timestamp()
        df_agg = df_agg.rename(columns={'Total_Neto':'y'})
        df_agg = df_agg.sort_values('ds')

    # Separar datos reales hasta fecha_inicio_pred (inclusive)
    df_train = df_agg[df_agg['ds'] <= pd.to_datetime(fecha_inicio_pred)].copy()
    df_test_start = df_train['ds'].max()
    
    # Generar fechas futuras para predicción
    if tipo_pred == "Diaria":
        fechas_futuras = pd.date_range(start=df_test_start + pd.Timedelta(days=1), periods=horizonte, freq='D')
    else:
        fechas_futuras = pd.date_range(start=df_test_start + pd.offsets.MonthBegin(1), periods=horizonte, freq='MS')

    if st.button("Generar predicción"):
        if modelo_sel == "Prophet":
            with st.spinner("Entrenando modelo Prophet..."):
                model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
                model.fit(df_train)

                futuro = pd.DataFrame({'ds': list(df_train['ds']) + list(fechas_futuras)})
                futuro = futuro.reset_index(drop=True)

                forecast = model.predict(futuro)

                # Mostrar gráfico con entrenamiento, real y predicción (solo fechas futuras)
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(title=f"Predicción de ventas netas para {unidad_ml} con Prophet", font_color='red')
                plotly_config_theme(fig1)
                st.plotly_chart(fig1, use_container_width=True)

        elif modelo_sel == "XGBoost":
            with st.spinner("Entrenando modelo XGBoost..."):
                # Crear features según tipo de predicción
                if tipo_pred == "Diaria":
                    df_train['dayofweek'] = df_train['ds'].dt.dayofweek
                    df_train['dayofmonth'] = df_train['ds'].dt.day
                    df_train['month'] = df_train['ds'].dt.month
                    df_train['year'] = df_train['ds'].dt.year

                    features = ['dayofweek', 'dayofmonth', 'month', 'year']
                else:
                    # Para mensual solo año y mes
                    df_train['month'] = df_train['ds'].dt.month
                    df_train['year'] = df_train['ds'].dt.year
                    features = ['month', 'year']

                X_train = df_train[features]
                y_train = df_train['y']

                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                model.fit(X_train, y_train)

                # Preparar dataframe para fechas futuras
                if tipo_pred == "Diaria":
                    df_future = pd.DataFrame({'ds': fechas_futuras})
                    df_future['dayofweek'] = df_future['ds'].dt.dayofweek
                    df_future['dayofmonth'] = df_future['ds'].dt.day
                    df_future['month'] = df_future['ds'].dt.month
                    df_future['year'] = df_future['ds'].dt.year
                    X_future = df_future[features]
                else:
                    df_future = pd.DataFrame({'ds': fechas_futuras})
                    df_future['month'] = df_future['ds'].dt.month
                    df_future['year'] = df_future['ds'].dt.year
                    X_future = df_future[features]

                preds_future = model.predict(X_future)

                # Unir datos reales y predicción
                df_pred_total = pd.concat([
                    df_train[['ds','y']],
                    pd.DataFrame({'ds': fechas_futuras, 'y': preds_future})
                ], ignore_index=True)

                # Gráfico de entrenamiento y predicción
                fig2 = px.line(title=f"Predicción XGBoost para {unidad_ml}")
                fig2.add_scatter(x=df_train['ds'], y=df_train['y'], mode='lines+markers', name='Entrenamiento')
                fig2.add_scatter(x=df_train['ds'], y=df_train['y'], mode='lines+markers', name='Real')
                fig2.add_scatter(x=fechas_futuras, y=preds_future, mode='lines+markers', name='Predicción')

                plotly_config_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)
