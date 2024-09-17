import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
from mlxtend.frequent_patterns import apriori, association_rules
import datetime
import re
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go



# Configuración de sesión
def login():
    st.sidebar.title("Inicio de sesión")
    username_input = st.sidebar.text_input("Usuario")
    password_input = st.sidebar.text_input("Contraseña", type="password")

    # Obtener las credenciales desde los secretos de Streamlit
    username_secret = st.secrets["login"]["username"]
    password_secret = st.secrets["login"]["password"]

    if st.sidebar.button("Iniciar sesión"):
        if username_input == username_secret and password_input == password_secret:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username_input
        else:
            st.sidebar.error("Nombre de usuario o contraseña incorrectos")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:

    # Mostrar el logo y el título
    st.image('./src/static/img/logo.png', width=200)  # Reemplaza 'path_to_your_logo.png' con la ruta a tu archivo de logo
    st.title('Análisis de Reglas de Asociación para Lulis')

    # Cargar el archivo CSV (asegúrate de subir el archivo con Streamlit)
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df

    # Filtrar por rango de fechas
    def filter_data_by_date(df, start_date, end_date):
        return df[(df['fecha'] >= pd.to_datetime(start_date)) & (df['fecha'] <= pd.to_datetime(end_date))]

    # Homologar productos
    def homologar_productos(df, umbral=90):
        productos = df['full_product_name'].unique()
        homologaciones = {}
        agrupados = []
        no_homologados = []

        for producto in productos:
            if producto in agrupados:
                continue

            matches = process.extract(producto, productos, scorer=fuzz.ratio, limit=len(productos))
            productos_similares = [match[0] for match in matches if match[1] >= umbral]

            if len(productos_similares) == 1:
                no_homologados.append(producto)
                continue

            nombre_homologado = productos_similares[0]
            homologaciones[nombre_homologado] = productos_similares
            agrupados.extend(productos_similares)

        def obtener_nombre_homologado(producto):
            for nombre_homologado, similares in homologaciones.items():
                if producto in similares:
                    return nombre_homologado
            return producto

        df['nombre_homologado'] = df['full_product_name'].apply(obtener_nombre_homologado)
        return df

    def limpiar_caracteres_y_repetidos(texto):
        # Eliminar cualquier carácter que no sea letra, número o espacio
        texto_limpio = re.sub(r'[^\w\s]', '', texto)
        texto_limpio = re.sub(r'\blulis\b', '', texto_limpio, flags=re.IGNORECASE)
        # Dividir el texto en palabras
        palabras = texto_limpio.split()
        # Eliminar palabras repetidas manteniendo el orden
        palabras_sin_repetir = []
        for palabra in palabras:
            if palabra not in palabras_sin_repetir:
                palabras_sin_repetir.append(palabra)
        # Volver a unir las palabras en una sola cadena
        return ' '.join(palabras_sin_repetir)
    
    from pyvis.network import Network
    import streamlit.components.v1 as components 

    
    def draw_network(rules, metric="lift", threshold=1):
        # Crear un grafo dirigido usando NetworkX
        G = nx.DiGraph()

        for _, row in rules.iterrows():
            if row[metric] >= threshold:
                G.add_edge(str(row['antecedents']), str(row['consequents']), weight=row[metric])

        # Obtener la posición de los nodos
        pos = nx.spring_layout(G, k=0.15, iterations=20)

        # Crear listas para las posiciones y los edges
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        # Crear las líneas para las aristas
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Crear los nodos
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Crear los círculos para los nodos
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',  # Aquí mostramos tanto los nodos como las etiquetas de texto
            text=[node for node in G.nodes()],  # Nombres dentro de los nodos
            textposition='middle center',  # Posición del texto en el centro del nodo
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Peso',
                    xanchor='left',
                    titleside='right'
                ),
            )
        )

        # Crear la figura con Plotly
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Red de productos basada en ' + metric,
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )

        # Mostrar el gráfico interactivo en Streamlit
        st.plotly_chart(fig)



    # Cargar los datos desde un archivo subido
    st.sidebar.header("Sube el archivo CSV con los datos")
    uploaded_file = st.sidebar.file_uploader("Subir archivo", type=["csv"])

    if uploaded_file is not None:
        # Cargar datos
        df = load_data(uploaded_file)

        # Filtrar por fechas
        st.sidebar.header("Filtro de Fechas")
        start_date = st.sidebar.date_input("Fecha de inicio", datetime.date(2023, 1, 1))
        end_date = st.sidebar.date_input("Fecha de fin", datetime.date(2024, 12, 31))
        
        filtered_data = filter_data_by_date(df, start_date, end_date)

        st.sidebar.header("Filtro de Tienda")
        tiendas_unicas = ['Todas'] + list(filtered_data['tienda'].unique())  # Añadir la opción 'Todas'
        tienda_seleccionada = st.sidebar.selectbox("Selecciona una tienda", tiendas_unicas)  # Crear el filtro de tienda

        # Aplicar el filtro de tienda solo si se selecciona una tienda específica
        if tienda_seleccionada != 'Todas':
            filtered_data = filtered_data[filtered_data['tienda'] == tienda_seleccionada]

        # Homologar los productos
        st.write("Homologando productos...")
        filtered_data['full_product_name'] = filtered_data['full_product_name'].apply(limpiar_caracteres_y_repetidos)
        df_homologado = homologar_productos(filtered_data)

        # Crear la canasta
        order_products = df_homologado
        basket = (order_products.groupby(['order_id', 'nombre_homologado'])['qty']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('order_id'))

        # Convertir las cantidades a valores binarios
        basket = basket.applymap(lambda x: 1 if x >= 1 else 0)

        # Aplicar el algoritmo apriori
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(x)))

        # Mostrar las reglas de asociación
        st.write("Reglas de Asociación")
        #st.dataframe(rules[['antecedents', 'consequents', 'Frecuencia', 'Probabilidad', 'Relacion']])
        rules2 = rules.copy()
        rules2.rename(columns = {"support": "Frecuencia", "confidence": "Probabilidad", "lift": "Relacion"}, inplace = True)

        st.markdown(
            """
            <div padding:10px; border-radius:5px;">
                <small>
                    <p><strong>Frecuencia:</strong> Es la proporción de transacciones que contienen un conjunto de productos en comparación con el total de transacciones. Representa la popularidad de un ítem o conjunto de ítems.</p>
                    <p><strong>Probabilidad:</strong> Es la probabilidad de que un producto sea comprado dado que otro producto ya ha sido comprado. Se calcula como la proporción de transacciones que contienen ambos productos (X e Y) con respecto a las que contienen solo X.</p>
                    <p><strong>Relación:</strong> Mide la relación entre la ocurrencia de ambos productos juntos y lo que se esperaría si fueran independientes. Un lift mayor a 1 indica una relación positiva, es decir, que los productos tienden a comprarse juntos más de lo esperado.</p>
                </small>
            </div>
            """, unsafe_allow_html=True
        )

        
        st.dataframe(rules2[['antecedents', 'consequents', 'Frecuencia', 'Probabilidad', 'Relacion']])

        # Mostrar los productos más frecuentes
        st.write("Top productos más frecuentes")
        top_products = frequent_itemsets.nlargest(10, 'support')
        top_products = top_products.sort_values(by='support', ascending=False) 
        fig = px.bar(top_products, x='itemsets', y='support',
                 title='Top 10 Productos Más Frecuentes',
                 labels={'itemsets': 'Productos', 'support': 'Soporte'},
                 color='support',
                 color_continuous_scale='Blues')

        # Ajustar la visualización
        fig.update_layout(xaxis_title='Productos',
                        yaxis_title='Soporte',
                        xaxis_tickangle=-45)

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)

         # Mostrar las gráficas
        st.header("Visualizaciones de Reglas de Asociación")

        # Gráfico de Red
        st.subheader("Gráfico de Red")
        draw_network(rules, metric="lift", threshold=1.5)

else:
    login()
