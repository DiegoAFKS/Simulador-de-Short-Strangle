import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go

# Función para obtener información de la acción
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get("shortName", "Nombre no disponible")
        sector = info.get("sector", "Sector no disponible")
        industry = info.get("industry", "Industria no disponible")
        return company_name, sector, industry
    except Exception:
        return None, None, None

# Función para calcular rendimientos con base 0
def calculate_returns(data):
    returns = (data["Close"] / data["Close"].iloc[0] - 1) * 100
    return returns

# Configuración de la app
st.title("Información y Rendimientos de Acciones")
st.sidebar.header("CONFIGURACIÓN DEL TICKER")

# Entrada del usuario: Ticker
ticker = st.sidebar.text_input("Ingresa el ticker de la acción para la estrategia:", value="AAPL").upper()

# Selección del período
period_years = st.sidebar.slider("Selecciona el período a analizar (en años):", min_value=1, max_value=10, value=3)

# Descargar datos históricos
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=period_years)

try:
    # Descargar datos de la acción
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No se encontraron datos para el ticker proporcionado.")
    else:
        # Obtener información de la acción
        company_name, sector, industry = get_stock_info(ticker)
        if company_name is None:
            st.error("El ticker ingresado no es válido.")
        else:
            # Mostrar información de la acción
            st.write(f"### Información de {ticker}")
            st.write(f"**Nombre de la Compañía:** {company_name}")
            st.write(f"**Sector:** {sector}")
            st.write(f"**Industria:** {industry}")

            # Calcular rendimientos
            data["Returns"] = calculate_returns(data)

            # Graficar rendimientos con base 0
            st.write(f"### Rendimientos de {ticker} en los últimos {period_years} años")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data["Returns"],
                mode="lines",
                name=f"Rendimientos ({ticker})",
                line=dict(color="blue")
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="black", annotation_text="Base 0")
            fig.update_layout(
                title=f"Rendimientos históricos de {ticker}",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Ocurrió un error al procesar los datos. Verifica el ticker ingresado.")


# Pie de página
st.sidebar.markdown("---")

# Configuración del Short Strangle
st.sidebar.header("Configuración del Short Strangle")

# Menú desplegable para Take Profit
take_profit_options = {
    "25% del crédito recibido": 0.25,
    "50% del crédito recibido": 0.50,
    "75% del crédito recibido": 0.75,
    "Al vencimiento": 1.0  # Se asume cierre automático
}
take_profit_label = st.sidebar.selectbox("Selecciona el Take Profit:", list(take_profit_options.keys()))
take_profit_percentage = take_profit_options[take_profit_label]

# Menú desplegable para Stop Loss
stop_loss_options = {
    "25% de la prima inicial": 0.25,
    "50% de la prima inicial": 0.50,
    "75% de la prima inicial": 0.75,
    "100% de la prima inicial": 1.0
}
stop_loss_label = st.sidebar.selectbox("Selecciona el Stop Loss:", list(stop_loss_options.keys()))
stop_loss_percentage = stop_loss_options[stop_loss_label]

# Mostrar las configuraciones seleccionadas
st.write(f"### Configuración del Short Strangle")
st.write(f"- **Take Profit:** {take_profit_label}")
st.write(f"- **Stop Loss:** {stop_loss_label}")


# Configuración inicial de Streamlit
st.title("Backtesting de Estrategia Short Strangle")
st.write("La estrategia busca el mejor precio Spot para abrir una posición de Short Strangle a 45 días")



st.sidebar.markdown("---")
st.sidebar.write("**Creado por D - A **")

# Descargar datos históricos
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=period_years)
data = yf.download(ticker, start=start_date, end=end_date)

# Calcular volatilidad histórica (HV)
data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
hv = data['Log Returns'].rolling(window=30).std() * np.sqrt(252)  # Anualizar HV
data['HV'] = hv

# Aproximar IV dinámica en función de la HV
iv_multiplier = 1.25  # Ajuste aproximado
data['IV'] = data['HV'] * iv_multiplier

# Calcular IVR basado en IV dinámica
data['IVR'] = (data['IV'] - data['IV'].rolling(window=252).min()) / (
    data['IV'].rolling(window=252).max() - data['IV'].rolling(window=252).min()) * 100

# Identificar rango estable +/-10% en los últimos mes y medio (45 días)
data['Range Center'] = data['Close'].rolling(window=45).mean()
data['Stable Range'] = ((data['Close'] >= 0.9 * data['Range Center']) & 
                        (data['Close'] <= 1.1 * data['Range Center']))

# Ajustar criterios
iv_threshold = 40  # Ajustado a 40% en lugar de 50%
data['IVR Criteria'] = data['IVR'] >= iv_threshold
data['Range Criteria'] = data['Stable Range']
data['Open Position Criteria'] = data['IVR Criteria'] & data['Range Criteria']

# Bucle para verificar y gestionar posiciones
positions = []
results = []
open_position = None  # Variable para rastrear la posición abierta
open_position_points = []  # Lista para registrar puntos de apertura de posiciones

for i, row in data.iterrows():
    if open_position:
        # Si hay una posición abierta, verificar TP, SL o vencimiento
        if row['Close'] <= open_position['TP'] or row['Close'] >= open_position['SL'] or i >= open_position['Expiration']:
            results.append({
                'Entry Date': open_position['Entry Date'],
                'Exit Date': i,
                'Entry Price': open_position['Entry Price'], 
                'Result': (row['Close'] - open_position['Entry Price']) * open_position['Contracts']
            })
            open_position = None  # Cerrar posición
    elif row['Open Position Criteria']:
        # Intentar abrir una nueva posición
        if open_position is None:  # Solo abrir si no hay otra posición
            expiration_date = i + pd.DateOffset(days=45)
            open_position = {
                'Entry Date': i,
                'Entry Price': row['Close'],
                'Contracts': 1,
                'TP': row['Close'] * (1 - take_profit_percentage),
                'SL': row['Close'] * (1 + stop_loss_percentage),
                'Expiration': expiration_date
            }
            positions.append(open_position)
            open_position_points.append((i, row['Close']))  # Registrar el punto de apertura


closed_position_points = [(result['Exit Date'], data.loc[result['Exit Date'], 'Close']) for result in results]


# Resumir resultados
total_positions = len(results)
profitable = sum(1 for r in results if r['Result'] > 0)
losses = total_positions - profitable
net_result = (sum(r['Result'] for r in results))*100

# Mostrar resumen de resultados
st.write("### Resumen de Resultados")
st.write(f"- **Total de posiciones abiertas:** {total_positions}")
st.write(f"- **Operaciones rentables:** {profitable}")
st.write(f"- **Operaciones negativas:** {losses}")
st.write(f"- **Resultado neto por las 100 opciones:** ${net_result:.2f}")

# Identificar las operaciones con mayor ganancia y mayor pérdida
if results:
    max_gain_operation = max(results, key=lambda x: x['Result'])  # Mayor ganancia
    max_loss_operation = min(results, key=lambda x: x['Result'])  # Mayor pérdida

    # Mostrar resultados
    st.write("### Operaciones Destacadas")
    st.write(f"- **Mayor Ganancia:** ${max_gain_operation['Result']:.2f} obtenida entre "
             f"{max_gain_operation['Entry Date']} y {max_gain_operation['Exit Date']}")
    st.write(f"- **Mayor Pérdida:** ${max_loss_operation['Result']:.2f} registrada entre "
             f"{max_loss_operation['Entry Date']} y {max_loss_operation['Exit Date']}")
else:
    st.write("No hay resultados disponibles para identificar operaciones destacadas.")

# Graficar la evolución del precio durante la operación más exitosa
if results:
    # Obtener la operación con mayor ganancia
    max_gain_operation = max(results, key=lambda x: x['Result'])

    # Extraer las fechas de entrada y salida
    entry_date = max_gain_operation['Entry Date']
    exit_date = max_gain_operation['Exit Date']

    # Extraer datos de la operación
    max_gain_data = data.loc[entry_date:exit_date].copy()

    # Calcular ganancia/pérdida acumulada
    entry_price = max_gain_operation['Entry Price']
    max_gain_data['Gain/Loss'] = max_gain_data['Close'] - entry_price

    # Crear la gráfica
    fig_max_gain = go.Figure()

    # Evolución del precio
    fig_max_gain.add_trace(go.Scatter(
        x=max_gain_data.index,
        y=max_gain_data['Close'],
        mode='lines',
        name='Evolución del Precio',
        line=dict(color='blue')
        
    ))
    


    # Puntos de ganancia
    fig_max_gain.add_trace(go.Scatter(
        x=max_gain_data[max_gain_data['Gain/Loss'] > 0].index,
        y=max_gain_data[max_gain_data['Gain/Loss'] > 0]['Close'],
        mode='markers',
        name='Ganancia',
        marker=dict(color='green', size=6, symbol='circle')
    ))

    # Puntos de pérdida
    fig_max_gain.add_trace(go.Scatter(
        x=max_gain_data[max_gain_data['Gain/Loss'] < 0].index,
        y=max_gain_data[max_gain_data['Gain/Loss'] < 0]['Close'],
        mode='markers',
        name='Pérdida',
        marker=dict(color='red', size=10, symbol='circle')
    ))

    # Configuración de la gráfica
    fig_max_gain.update_layout(
        title="Evolución del Precio Durante la Operación Más Exitosa",
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        template="plotly_white"
    )

    # Mostrar la gráfica
    st.plotly_chart(fig_max_gain, use_container_width=True)

# Graficar la evolución del precio durante la operación con mayor pérdida
if results:
    # Obtener la operación con mayor pérdida
    max_loss_operation = min(results, key=lambda x: x['Result'])

    # Extraer las fechas de entrada y salida
    entry_date = max_loss_operation['Entry Date']
    exit_date = max_loss_operation['Exit Date']

    # Extraer datos de la operación
    max_loss_data = data.loc[entry_date:exit_date].copy()

    # Calcular ganancia/pérdida acumulada
    entry_price = max_loss_operation['Entry Price']
    max_loss_data['Gain/Loss'] = max_loss_data['Close'] - entry_price

    # Crear la gráfica
    fig_max_loss = go.Figure()

    # Evolución del precio
    fig_max_loss.add_trace(go.Scatter(
        x=max_loss_data.index,
        y=max_loss_data['Close'],
        mode='lines',
        name='Evolución del Precio',
        line=dict(color='blue')
    ))

    # Puntos de ganancia
    fig_max_loss.add_trace(go.Scatter(
        x=max_loss_data[max_loss_data['Gain/Loss'] > 0].index,
        y=max_loss_data[max_loss_data['Gain/Loss'] > 0]['Close'],
        mode='markers',
        name='Ganancia',
        marker=dict(color='green', size=6, symbol='circle')
    ))

    # Puntos de pérdida
    fig_max_loss.add_trace(go.Scatter(
        x=max_loss_data[max_loss_data['Gain/Loss'] < 0].index,
        y=max_loss_data[max_loss_data['Gain/Loss'] < 0]['Close'],
        mode='markers',
        name='Pérdida',
        marker=dict(color='red', size=6, symbol='circle')
    ))

    # Configuración de la gráfica
    fig_max_loss.update_layout(
        title="Evolución del Precio Durante la Operación con Mayor Pérdida",
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        template="plotly_white"
    )

    # Mostrar la gráfica
    st.plotly_chart(fig_max_loss, use_container_width=True)

# Graficar resultados acumulados
if results:
    # Multiplicar cada resultado por 100 para reflejar el tamaño del contrato
    cumulative_results = np.cumsum([r['Result'] * 100 for r in results])  # Acumulado de resultados

    fig_results = go.Figure()
    fig_results.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_results) + 1)),
        y=cumulative_results,
        mode='lines',
        name='Resultado Monetario Acumulado ($)',
        line=dict(color='blue')
    ))
    fig_results.update_layout(
        title="Resultado Monetario Acumulado",
        xaxis_title="Número de Operaciones",
        yaxis_title="Resultado Acumulado ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_results, use_container_width=True)
else:
    st.write("No se generaron resultados acumulados porque no se abrieron posiciones.")




# Visualizar evolución del IVR
fig_ivr = go.Figure()
fig_ivr.add_trace(go.Scatter(
    x=data.index,
    y=data['IVR'],
    mode='lines',
    name='IVR'
))
fig_ivr.add_hline(y=iv_threshold, line_dash="dot", line_color="green", annotation_text=f"Criterio IVR >= {iv_threshold}")
fig_ivr.update_layout(
    title="Evolución del IVR",
    xaxis_title="Fecha",
    yaxis_title="IVR (%)",
    template="plotly_white"
)
st.plotly_chart(fig_ivr, use_container_width=True)

# Visualizar rango estable con puntos de apertura de posiciones
fig_range = go.Figure()
fig_range.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Precio de Cierre'
))
fig_range.add_trace(go.Scatter(
    x=data.index,
    y=data['Range Center'],
    mode='lines',
    name='Centro del Rango (+/-10%)',
    line=dict(dash='dot')
))
fig_range.add_trace(go.Scatter(
    x=data.index,
    y=0.9 * data['Range Center'],
    mode='lines',
    name='Límite Inferior del Rango',
    line=dict(dash='dash')
))
fig_range.add_trace(go.Scatter(
    x=data.index,
    y=1.1 * data['Range Center'],
    mode='lines',
    name='Límite Superior del Rango',
    line=dict(dash='dash')
))

# Agregar puntos donde se abrieron posiciones
for point in open_position_points:
    fig_range.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        name='Apertura de Posición',
        marker=dict(color='blue', size=6, symbol='circle')
    ))

for point in closed_position_points:
    fig_range.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        name='Cierre de Posición',
        marker=dict(color='red', size=6, symbol='circle')
    ))


fig_range.update_layout(
    title="Evolución del Precio y Rango Estable con Aperturas de Posición",
    xaxis_title="Fecha",
    yaxis_title="Precio ($)",
    template="plotly_white"
)
st.plotly_chart(fig_range, use_container_width=True)

