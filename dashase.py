import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

st.set_page_config(page_title="Dashboard Ventas Interactivo", layout="wide")

# --- Estilos CSS personalizados ---
st.markdown("""
<style>
/* Fondo general con azul medio */
[data-testid="stAppViewContainer"] {
    background: #4a90e2; /* azul medio */
    color: #003049;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Encabezados */
h1, h2, h3, h4, h5 {
    color: #2e7d32;  /* verde oscuro */
    font-weight: 700;
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background: #a5d6a7;  /* verde claro */
    color: #004d40;
    font-weight: 600;
}

/* Métricas */
[data-testid="stMetricValue"] {
    color: #ff6f00; /* naranja vibrante */
    font-weight: 700;
}

/* Botones, selects y multiselect */
.stSelectbox, .stMultiselect, .stButton {
    border-radius: 10px;
    background-color: #ffb74d; /* naranja claro */
    color: #3e2723;
    font-weight: 600;
}

/* Tabla */
.stDataFrame div[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #2e7d32; /* verde oscuro */
    background-color: #dcedc8; /* verde muy claro */
    color: #1b5e20;
}

/* Pie chart texto y otros textos dentro gráficos Plotly */
svg text, .xtick, .ytick {
    fill: black !important;
    font-weight: 700;
}

/* Títulos gráficos */
.plotly .main-svg .g-gtitle {
    fill: black !important;
    font-weight: 700;
}

/* Mejorar scroll en tablas */
div[data-testid="stDataFrame"] > div > div {
    scrollbar-color: #2e7d32 #dcedc8;
    scrollbar-width: thin;
}

/* Scroll personalizado para navegadores Webkit */
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar {
    width: 8px;
}
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar-track {
    background: #dcedc8;
}
div[data-testid="stDataFrame"] > div > div::-webkit-scrollbar-thumb {
    background-color: #2e7d32;
    border-radius: 10px;
    border: 2px solid #a5d6a7;
}
</style>
""", unsafe_allow_html=True)

# --- Función para aplicar tema a gráficos Plotly ---
def plotly_config_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(255, 204, 203, 0.4)',  # rojo suave con transparencia
        paper_bgcolor='rgba(255, 204, 203, 0.4)',
        font=dict(color='black', family='Segoe UI', size=14),
        title_font=dict(color='black', size=18, family='Segoe UI'),
        legend_font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.15)', zerolinecolor='rgba(0,0,0,0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.15)', zerolinecolor='rgba(0,0,0,0.3)'),
    )

# --- Ruta base donde están los logos ---
ruta_logos = r"C:\Users\yyang\Downloads"

def cargar_logo(nombre_archivo):
    ruta = os.path.join(ruta_logos, nombre_archivo)
    try:
        return Image.open(ruta)
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen {nombre_archivo}: {e}")
        return None

# --- Mostrar logo Asecom en encabezado ---
logo_asecom = cargar_logo("Asecom.png")
if logo_asecom:
    st.image(logo_asecom, width=150, caption="Logo Asecom", use_column_width=False)

st.markdown("# 📊 Dashboard Interactivo de Ventas 2025")

# --- Diccionario para clasificación automática según descripción ---
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

@st.cache_data
def cargar_datos():
    df = pd.read_excel('2025.xlsx')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha'])
    df['IVA'] = (df['Valor'] * df['Cantidad']) - df['Total']
    
    df['Categoría'] = df['Descripción'].apply(clasificar_producto)
    return df

df = cargar_datos()

# --- Filtros laterales ---
st.sidebar.header("🔎 Filtros (opcionales)")

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

# --- Aplicar filtros ---
df_filtrado = df[
    (df['Fecha'] >= pd.to_datetime(rango_fecha[0])) &
    (df['Fecha'] <= pd.to_datetime(rango_fecha[1]))
]

for columna, valores in filtros.items():
    if valores:
        df_filtrado = df_filtrado[df_filtrado[columna].isin(valores)]

# --- Mostrar logos según filtro unidad ---
if filtros["UNIDAD"]:
    unidades = filtros["UNIDAD"]
    logos_mostrados = []
    for unidad in unidades:
        if unidad.lower() == "GO CAFE" and "Cafe Go.png" not in logos_mostrados:
            logo_cafe = cargar_logo("Cafe Go.png")
            if logo_cafe:
                st.image(logo_cafe, width=180, caption="Cafe Go")
                logos_mostrados.append("Cafe Go.png")
        elif unidad.lower() == "pizza go" and "Pizza go.png" not in logos_mostrados:
            logo_pizza = cargar_logo("Pizza go.png")
            if logo_pizza:
                st.image(logo_pizza, width=180, caption="Pizza Go")
                logos_mostrados.append("Pizza go.png")
        elif unidad.lower() == "cafeteria" and "Cafeteria.png" not in logos_mostrados:
            logo_cafeteria = cargar_logo("Cafeteria.png")
            if logo_cafeteria:
                st.image(logo_cafeteria, width=180, caption="Cafetería")
                logos_mostrados.append("Cafeteria.png")

# --- Selector de KPI ---
kpi = st.selectbox("📌 Selecciona un KPI para visualizar", [
    "Total Vendido",
    "Impuesto IVA",
    "Productos más vendidos (general)",
    "Productos más vendidos por unidad"
])

# --- Mostrar KPI y gráfico correspondiente ---
if kpi == "Total Vendido":
    df_filtrado["Tipo Movto."] = df_filtrado["Tipo Movto."].astype(str).str.strip()

    total_venta = df_filtrado[df_filtrado["Tipo Movto."] == "VT-VENTA"]["Total"].sum()
    total_devolucion = df_filtrado[df_filtrado["Tipo Movto."] == "RV-DEVOLUCION DE VENTAS"]["Total"].sum()
    total_compras_credito = df_filtrado[df_filtrado["Tipo Movto."] == "CP-COMPRAS AL CREDITO (PROVEEDORES)"]["Total"].sum()

    total_vendido_neto = total_venta - total_devolucion - total_compras_credito

    st.metric("💰 Total Vendido Neto", f"Q{total_vendido_neto:,.2f}")

    fig = px.histogram(
        df_filtrado[df_filtrado["Tipo Movto."].isin([
            "VT-VENTA", 
            "RV-DEVOLUCION DE VENTAS", 
            "CP-COMPRAS AL CREDITO (PROVEEDORES)"
        ])],
        x="UNIDAD", y="Total", histfunc="sum", color="Tipo Movto.",
        title="Total por Unidad y Tipo de Movimiento"
    )
    plotly_config_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

elif kpi == "Impuesto IVA":
    iva = df_filtrado["IVA"].sum()
    st.metric("🧾 Total IVA", f"Q{iva:,.2f}")
    fig = px.histogram(df_filtrado, x="UNIDAD", y="IVA", histfunc="sum", title="IVA por Unidad")
    plotly_config_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

elif kpi == "Productos más vendidos (general)":
    top_productos = df_filtrado.groupby("Descripción")["Cantidad"].sum().sort_values(ascending=False).head(10)
    fig = px.bar(top_productos, x=top_productos.values, y=top_productos.index, orientation='h',
                 title="📦 Productos Más Vendidos (General)", labels={"x": "Cantidad"})
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
                title=f"🥧 Top Productos más vendidos en {unidad}"
            )
            plotly_config_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Por favor selecciona al menos una UNIDAD para ver este gráfico.")

# --- Mostrar tabla resultante ---
st.markdown("### 📄 Registros filtrados")
st.dataframe(df_filtrado, use_container_width=True)

# --- NUEVO APARTADO: COMPRADORES ---
st.markdown("---")
st.subheader("🧍 Compradores Destacados")

ventas_df = df_filtrado[df_filtrado["Tipo Movto."].isin([
    "VT-VENTA",
    "RV-DEVOLUCION DE VENTAS",
    "CP-COMPRAS AL CREDITO (PROVEEDORES)"
])]

total_venta = ventas_df[ventas_df["Tipo Movto."] == "VT-VENTA"]["Total"].sum()
total_devolucion = ventas_df[ventas_df["Tipo Movto."] == "RV-DEVOLUCION DE VENTAS"]["Total"].sum()
total_compras_credito = ventas_df[ventas_df["Tipo Movto."] == "CP-COMPRAS AL CREDITO (PROVEEDORES)"]["Total"].sum()

total_vendido_neto = total_venta - total_devolucion - total_compras_credito

ventas_only = ventas_df[ventas_df["Tipo Movto."] == "VT-VENTA"]

top_clientes = ventas_only.groupby('Nombre')['Total'].sum().reset_index()
top_clientes = top_clientes.sort_values(by='Total', ascending=False).head(10)

top_total = top_clientes['Total'].sum()
otros_total = total_vendido_neto - top_total

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
    '#2e7d32' if nombre in seleccion else 'lightgrey'
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
    hole=0.3
)
fig_pie.update_traces(marker=dict(colors=plot_colors), textposition='inside', textinfo='percent+label')

plotly_config_theme(fig_pie)
st.plotly_chart(fig_pie, use_container_width=True)
