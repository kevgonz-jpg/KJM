import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Configuración de rutas relativas y absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS_DIR = os.path.join(BASE_DIR, 'sistema_procesamiento', 'datos')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'sistema_procesamiento', 'outputs')
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')
PLOTS_DIR = os.path.join(DASHBOARD_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
# 1. CARGA DE DATOS COMUNES
print("Cargando datos principales de bonos y tasas...")
# Carga de info estática
df_estatica = pd.read_csv(os.path.join(DATOS_DIR, 'bonos_info_estatica.csv'), parse_dates=['dueDate'])
# Carga de operaciones TFIT (Mercado) para preprocesamiento
dbs_tfit = [
    os.path.join(DATOS_DIR, 'BVC_Bonos_2015_2018.db'),
    os.path.join(DATOS_DIR, 'BVC_Bonos_2019_2022.db'),
    os.path.join(DATOS_DIR, 'BVC_Bonos_2023_2026.db')
]
fragmentos_tfit = []
for db in dbs_tfit:
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        df_temp = pd.read_sql_query("SELECT * FROM operaciones", conn)
        conn.close()
        fragmentos_tfit.append(df_temp)
df_tfit_raw = pd.concat(fragmentos_tfit, ignore_index=True)
df_tfit_raw['fecha_neg'] = pd.to_datetime(df_tfit_raw['fecha_neg'])
# Preprocesamiento de TFIT
# VWAP
grp_tfit = df_tfit_raw.groupby(['fecha_neg', 'nemotecnico'])
tasa_vwap = ((df_tfit_raw['rate'] * df_tfit_raw['volume']).groupby([df_tfit_raw['fecha_neg'], df_tfit_raw['nemotecnico']]).sum() / grp_tfit['volume'].sum()).reset_index(name='tasa_vwap')
volumen_dia = grp_tfit['volume'].sum().reset_index(name='volumen_dia')
df_pre = tasa_vwap.merge(volumen_dia, on=['fecha_neg', 'nemotecnico'])
df_pre = df_pre.merge(df_estatica[['nemotecnico', 'dueDate', 'couponRate']], on='nemotecnico', how='left')
df_pre['tau'] = (df_pre['dueDate'] - df_pre['fecha_neg']).dt.days / 365.0
df_pre = df_pre[df_pre['tau'] > 0].copy()
df_pre = df_pre.sort_values(by=['fecha_neg', 'tau']).reset_index(drop=True)
print(f"[OK] TFIT Preprocesado: {len(df_pre):,} registros.")
# -----------------------------------------------------------------------------
# GRÁFICO 1: Evolución de Yields de Mercado (VWAP) 2016
# -----------------------------------------------------------------------------
print("Generando Grafico 1 (Yields de Mercado 2016)...")
df_viz1 = df_pre[df_pre['fecha_neg'].dt.year == 2016].copy() 
df_viz1['fecha_str'] = df_viz1['fecha_neg'].dt.strftime('%Y-%m-%d')
df_viz1 = df_viz1.sort_values('fecha_neg')
x_range = [0, df_viz1['tau'].max() * 1.05]
y_range = [0, df_viz1['tasa_vwap'].max() * 1.1]
fig1 = px.scatter(
    df_viz1,
    x='tau',
    y='tasa_vwap',
    animation_frame='fecha_str',
    animation_group='nemotecnico',
    color='nemotecnico',
    hover_name='nemotecnico',
    range_x=x_range,
    range_y=y_range,
    labels={
        'tau': 'Plazo Residual (Años)',
        'tasa_vwap': 'Rendimiento (VWAP %)',
        'fecha_str': 'Fecha de Negociación'
    },
    title='Curva TFIT: Evolución de Yields de Mercado (VWAP) - Año 2016',
    template='plotly_dark'
)
fig1.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')))
fig1.update_layout(
    height=600,
    margin=dict(t=80, b=80, l=60, r=40),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x': 0.1, 'y': -0.15,
        'xanchor': 'right', 'yanchor': 'top',
        'direction': 'left',
        'buttons': [
            {'label': '▶ Play', 'method': 'animate',
             'args': [None, {'frame': {'duration': 200, 'redraw': False}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)
plot1_path = os.path.join(PLOTS_DIR, 'chart1.html')
fig1.write_html(plot1_path, include_plotlyjs='cdn')
print(f"[OK] Grafico 1 guardado en: {plot1_path}")
# -----------------------------------------------------------------------------
# GRÁFICO 2: Validación Integral (último gráfico en experimentación_curva...)
# -----------------------------------------------------------------------------
print("Generando Grafico 2 (Validacion Integral Multiorigen)...")
# Carga de TIB y IBR
df_tib_raw = pd.read_csv(os.path.join(DATOS_DIR, 'TIB.csv'))
df_tib = pd.DataFrame()
df_tib['fecha_neg']   = pd.to_datetime(df_tib_raw['Periodo(MMM DD, AAAA)'], format='%Y/%m/%d', errors='coerce')
df_tib['tasa_vwap']   = df_tib_raw['Tasa interbancaria (TIB)'].astype(float)
df_tib['tau']         = 1 / 365
df_tib['nemotecnico'] = 'TIB (Overnight)'
df_tib['fecha_str']   = df_tib['fecha_neg'].dt.strftime('%Y-%m-%d')
df_ibr_raw = pd.read_csv(os.path.join(DATOS_DIR, 'IBR.csv'))
MAPA_IBR = {
    'Indicador Bancario de Referencia (IBR) overnight, nominal': 1/365,
    'Indicador Bancario de Referencia (IBR) a 1 mes, nominal':   1/12,
    'Indicador Bancario de Referencia (IBR) a 3 meses, nominal': 3/12,
    'Indicador Bancario de Referencia (IBR) a 6 meses, nominal': 6/12,
    'Indicador Bancario de Referencia (IBR) a 12 meses, nominal': 12/12,
}
NOMBRES_CORTOS = {
    'Indicador Bancario de Referencia (IBR) overnight, nominal': 'IBR Overnight',
    'Indicador Bancario de Referencia (IBR) a 1 mes, nominal':   'IBR 1M',
    'Indicador Bancario de Referencia (IBR) a 3 meses, nominal': 'IBR 3M',
    'Indicador Bancario de Referencia (IBR) a 6 meses, nominal': 'IBR 6M',
    'Indicador Bancario de Referencia (IBR) a 12 meses, nominal':'IBR 12M',
}
df_ibr_long = df_ibr_raw.melt(
    id_vars='Periodo(MMM DD, AAAA)',
    value_vars=list(MAPA_IBR.keys()),
    var_name='plazo_nombre',
    value_name='tasa_vwap'
).dropna(subset=['tasa_vwap']).copy()
df_ibr_long['tau']         = df_ibr_long['plazo_nombre'].map(MAPA_IBR)
df_ibr_long['nemotecnico'] = df_ibr_long['plazo_nombre'].map(NOMBRES_CORTOS)
df_ibr_long['fecha_neg']   = pd.to_datetime(df_ibr_long['Periodo(MMM DD, AAAA)'], format='%Y/%m/%d', errors='coerce')
df_ibr_long['fecha_str']   = df_ibr_long['fecha_neg'].dt.strftime('%Y-%m-%d')
# Carga TCO
dbs_tco = [
    os.path.join(DATOS_DIR, 'BVC_Bonos_TCO_2015_2018.db'),
    os.path.join(DATOS_DIR, 'BVC_Bonos_TCO_2019_2022.db'),
    os.path.join(DATOS_DIR, 'BVC_Bonos_TCO_2023_2026.db')
]
fragmentos_tco = []
for db in dbs_tco:
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        df_temp = pd.read_sql_query("SELECT * FROM operaciones", conn)
        conn.close()
        fragmentos_tco.append(df_temp)
df_cot_raw = pd.concat(fragmentos_tco, ignore_index=True)
df_cot_raw['fecha_neg'] = pd.to_datetime(df_cot_raw['fecha_neg'])
def parsear_nemo_cot_local(nemo: str) -> pd.Timestamp | None:
    try:
        dia  = int(nemo[6:8])
        mes  = int(nemo[8:10])
        anio = 2000 + int(nemo[10:12])
        return pd.Timestamp(anio, mes, dia)
    except Exception:
        return None
df_cot_raw['dueDate'] = df_cot_raw['nemotecnico'].apply(parsear_nemo_cot_local)
df_cot_raw = df_cot_raw[df_cot_raw['dueDate'].notna()].copy()
df_cot_raw['tau'] = (df_cot_raw['dueDate'] - df_cot_raw['fecha_neg']).dt.days / 365.0
df_cot_raw = df_cot_raw[df_cot_raw['tau'] > 0].copy()
# VWAP TCO
grp_tco = df_cot_raw.groupby(['fecha_neg', 'nemotecnico'])
tasa_vwap_cot = ((df_cot_raw['rate'] * df_cot_raw['volume']).groupby([df_cot_raw['fecha_neg'], df_cot_raw['nemotecnico']]).sum() / grp_tco['volume'].sum()).reset_index(name='tasa_vwap')
volumen_dia_cot = grp_tco['volume'].sum().reset_index(name='volumen_dia')
tau_dia_cot = ((df_cot_raw['tau'] * df_cot_raw['volume']).groupby([df_cot_raw['fecha_neg'], df_cot_raw['nemotecnico']]).sum() / grp_tco['volume'].sum()).reset_index(name='tau')
due_dia_cot = grp_tco['dueDate'].first().reset_index()
df_cot_vwap = tasa_vwap_cot.merge(volumen_dia_cot, on=['fecha_neg', 'nemotecnico']).merge(tau_dia_cot, on=['fecha_neg', 'nemotecnico']).merge(due_dia_cot, on=['fecha_neg', 'nemotecnico'])
# Outputs de Bootstrapping y Panel
df_discreto = pd.read_csv(os.path.join(OUTPUTS_DIR, 'curvas_discretas_historico.csv'), parse_dates=['fecha'])
df_panel = pd.read_csv(os.path.join(OUTPUTS_DIR, 'curvas_panel_historico.csv'), parse_dates=['fecha'])
# Preparar fechas de animación (cada 10 días para fluidez y tamaño razonable de archivo)
fechas_animacion = sorted(df_pre['fecha_neg'].unique()[::10])
fecha_min = df_pre['fecha_neg'].min()
fecha_max = df_pre['fecha_neg'].max()
def filtrar_fechas(df, col_fecha='fecha'):
    return df[
        (df[col_fecha].isin(fechas_animacion)) &
        (df[col_fecha] >= fecha_min) &
        (df[col_fecha] <= fecha_max)
    ].copy()
# A. Mercado (VWAP)
df_v_mercado = filtrar_fechas(df_pre, col_fecha='fecha_neg').rename(columns={'fecha_neg': 'fecha', 'tasa_vwap': 'tasa'})
df_v_mercado['tasa']   = df_v_mercado['tasa'] / 100.0
df_v_mercado['fuente'] = 'Mercado (VWAP)'
df_v_mercado['label']  = df_v_mercado['nemotecnico']
# B. Spot Discreto
df_v_discreto = filtrar_fechas(df_discreto).rename(columns={'r_spot': 'tasa'})
df_v_discreto['fuente'] = 'Spot (Bootstrapping)'
df_v_discreto['label']  = df_v_discreto['nemotecnico']
# C. Panel Interpolado (wide → long)
df_v_panel = filtrar_fechas(df_panel).melt(id_vars='fecha', var_name='nodo', value_name='tasa').dropna(subset=['tasa'])
df_v_panel['tau']    = df_v_panel['nodo'].str.extract(r'(\d+)').astype(float)
df_v_panel['fuente'] = 'Curva Interpolada'
df_v_panel['label']  = df_v_panel['nodo'].str.replace('r_', '', regex=False).str.replace('yr', 'Y', regex=False)
# D. TIB
df_v_tib = filtrar_fechas(df_tib, col_fecha='fecha_neg').rename(columns={'fecha_neg': 'fecha', 'tasa_vwap': 'tasa'})
df_v_tib['tasa']   = df_v_tib['tasa'] / 100.0
df_v_tib['fuente'] = 'TIB (Overnight)'
df_v_tib['label']  = 'TIB'
# E. IBR
df_v_ibr = filtrar_fechas(df_ibr_long, col_fecha='fecha_neg').rename(columns={'fecha_neg': 'fecha', 'tasa_vwap': 'tasa'})
df_v_ibr['tasa']   = df_v_ibr['tasa'] / 100.0
df_v_ibr['fuente'] = 'IBR'
df_v_ibr['label']  = df_v_ibr['nemotecnico']
# F. TCO
df_v_cot = filtrar_fechas(df_cot_vwap, col_fecha='fecha_neg').rename(columns={'fecha_neg': 'fecha', 'tasa_vwap': 'tasa'})
df_v_cot['tasa']   = df_v_cot['tasa'] / 100.0
df_v_cot['fuente'] = 'TCO (Cero-Cupón)'
df_v_cot['label']  = df_v_cot['nemotecnico']
# Combinar
COLS = ['fecha', 'tau', 'tasa', 'fuente', 'label']
df_completo_v2 = pd.concat([
    df_v_mercado[COLS],
    df_v_discreto[COLS],
    df_v_panel[COLS],
    df_v_tib[COLS],
    df_v_ibr[COLS],
    df_v_cot[COLS],
], ignore_index=True)
df_completo_v2['fecha_str'] = df_completo_v2['fecha'].dt.strftime('%Y-%m-%d')
df_completo_v2['tasa_pct']  = (df_completo_v2['tasa'] * 100).round(4)
df_completo_v2 = df_completo_v2.sort_values(['fecha_str', 'fuente', 'tau']).reset_index(drop=True)
fechas_str = sorted(df_completo_v2['fecha_str'].unique())
# Construcción de Traces y Frames
COLORES = {
    'Mercado (VWAP)':       '#636EFA',
    'Spot (Bootstrapping)': '#EF553B',
    'Curva Interpolada':    '#00CC96',
    'TIB (Overnight)':      '#AB63FA',
    'IBR':                  '#FFA500',
    'TCO (Cero-Cupón)':     '#FF6B9D',
}
SIMBOLOS = {
    'Mercado (VWAP)':       'circle',
    'Spot (Bootstrapping)': 'diamond',
    'Curva Interpolada':    'cross',
    'TIB (Overnight)':      'star',
    'IBR':                  'triangle-up',
    'TCO (Cero-Cupón)':     'square',
}
TAMANOS = {
    'Mercado (VWAP)':       7,
    'Spot (Bootstrapping)': 10,
    'Curva Interpolada':    8,
    'TIB (Overnight)':      14,
    'IBR':                  12,
    'TCO (Cero-Cupón)':     11,
}
FUENTES_SCATTER = ['Mercado (VWAP)', 'Spot (Bootstrapping)', 'TIB (Overnight)', 'IBR', 'TCO (Cero-Cupón)']
FUENTES_LINEA = ['Curva Interpolada']
def build_traces(fecha_str: str) -> list:
    traces = []
    df_f = df_completo_v2[df_completo_v2['fecha_str'] == fecha_str]
    for fuente in FUENTES_SCATTER:
        df_s = df_f[df_f['fuente'] == fuente].sort_values('tau')
        if df_s.empty:
            traces.append(go.Scatter(x=[], y=[], mode='markers', name=fuente, marker=dict(color=COLORES[fuente], symbol=SIMBOLOS[fuente], size=TAMANOS[fuente]), showlegend=True))
        else:
            traces.append(go.Scatter(
                x=df_s['tau'], y=df_s['tasa_pct'], mode='markers', name=fuente,
                marker=dict(color=COLORES[fuente], symbol=SIMBOLOS[fuente], size=TAMANOS[fuente], opacity=0.85, line=dict(width=0.5, color='white')),
                customdata=df_s[['label', 'tasa_pct']].values,
                hovertemplate='<b>%{customdata[0]}</b><br>Plazo: %{x:.3f} años<br>Tasa: %{customdata[1]:.4f}%<br><i>'+fuente+'</i><extra></extra>',
                showlegend=True
            ))
    for fuente in FUENTES_LINEA:
        df_l = df_f[df_f['fuente'] == fuente].sort_values('tau')
        if df_l.empty:
            traces.append(go.Scatter(x=[], y=[], mode='lines+markers', name=fuente, line=dict(color=COLORES[fuente], width=2), showlegend=True))
        else:
            traces.append(go.Scatter(
                x=df_l['tau'], y=df_l['tasa_pct'], mode='lines+markers', name=fuente,
                line=dict(color=COLORES[fuente], width=2.5),
                marker=dict(color=COLORES[fuente], symbol=SIMBOLOS[fuente], size=TAMANOS[fuente], opacity=0.9),
                customdata=df_l[['label', 'tasa_pct']].values,
                hovertemplate='<b>%{customdata[0]}</b><br>Plazo: %{x:.0f} años<br>Tasa: %{customdata[1]:.4f}%<br><i>Curva Interpolada</i><extra></extra>',
                showlegend=True
            ))
    return traces
fig2 = go.Figure(data=build_traces(fechas_str[0]))
fig2.frames = [go.Frame(data=build_traces(fecha_str), name=fecha_str) for fecha_str in fechas_str]
# Configuración del Layout para Gráfico 2
NODOS_INTERPOLACION = [1, 2, 3, 5, 7, 10, 15, 20, 30]
fig2.update_layout(
    template='plotly_dark',
    title=dict(
        text='Validación Integral: Mercado · Bootstrapping · Interpolación · TIB · IBR · TCO',
        font=dict(size=14),
        x=0.5, xanchor='center'
    ),
    xaxis=dict(
        title='Plazo (Años)', range=[-0.3, 32],
        gridcolor='rgba(255,255,255,0.08)', zeroline=False,
    ),
    yaxis=dict(
        title='Tasa (%)', range=[3, 17],
        gridcolor='rgba(255,255,255,0.08)', ticksuffix='%', zeroline=False,
    ),
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        font=dict(size=10), bgcolor='rgba(0,0,0,0.3)'
    ),
    hovermode='closest',
    margin=dict(t=110, b=90, l=60, r=40),
    updatemenus=[dict(
        type='buttons', showactive=False, x=0.1, y=-0.13,
        xanchor='right', yanchor='top', direction='left',
        buttons=[
            dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label='⏸ Pausa', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )],
    sliders=[dict(
        active=0,
        steps=[
            dict(args=[[f], dict(frame=dict(duration=100, redraw=True), mode='immediate', transition=dict(duration=0))], label=f, method='animate')
            for f in fechas_str
        ],
        x=0.1, y=-0.07, len=0.9,
        currentvalue=dict(prefix='Fecha: ', font=dict(size=13), visible=True, xanchor='center'),
        transition=dict(duration=0), pad=dict(b=10, t=30),
    )]
)
# Añadir vlines estándar
for nodo in NODOS_INTERPOLACION:
    fig2.add_vline(
        x=nodo, line_width=1, line_dash='dash', line_color='rgba(255,255,255,0.12)',
        annotation_text=f'{nodo}Y', annotation_position='top left',
        annotation_font_size=9, annotation_font_color='rgba(255,255,255,0.35)'
    )
for etiqueta, x_val in {'1d': 1/365, '1M': 1/12, '3M': 3/12, '6M': 6/12, '12M': 1.0}.items():
    fig2.add_vline(
        x=x_val, line_width=1, line_dash='dot', line_color='rgba(255,165,0,0.35)',
        annotation_text=etiqueta, annotation_position='top right',
        annotation_font_size=8, annotation_font_color='#FFA500'
    )
plot2_path = os.path.join(PLOTS_DIR, 'chart2.html')
fig2.write_html(plot2_path, include_plotlyjs='cdn')
print(f"[OK] Grafico 2 guardado en: {plot2_path}")
# -----------------------------------------------------------------------------
# GRÁFICO 3: Superficie 3D (Limpieza_TESCOL_y_Dinamica.ipynb)
# -----------------------------------------------------------------------------
print("Generando Grafico 3 (Superficie 3D de Curva de Rendimientos)...")
df_3d_panel = pd.read_csv(os.path.join(BASE_DIR, 'TES_colombia', 'curvas_panel_historico.csv'))
vencimientos = [1, 2, 3, 5, 7, 10, 15]
columnas_bonos = ["r_1yr", "r_2yr", "r_3yr", "r_5yr", "r_7yr", "r_10yr", "r_15yr"]
df_plot = df_3d_panel.copy()
df_plot["fecha"] = pd.to_datetime(df_plot["fecha"], errors="coerce")
df_plot = df_plot.dropna(subset=["fecha"]).sort_values("fecha")
# Muestra temporal (cada 2 días) para que la renderización 3D en el navegador sea muy fluida
step = 2
fechas = df_plot["fecha"].iloc[::step]
matriz_rendimientos = df_plot[columnas_bonos].iloc[::step].values
fig3 = go.Figure(data=[go.Surface(
    z=matriz_rendimientos,
    x=vencimientos,
    y=fechas,
    colorscale="Viridis",
    colorbar_title="Tasa (%)"
)])
fig3.update_layout(
    template='plotly_dark',
    title=dict(
        text="Evolución de la Curva de Rendimientos 3D (TES Colombia)",
        font=dict(size=14),
        x=0.5, xanchor='center'
    ),
    autosize=True,
    height=650,
    margin=dict(l=50, r=50, b=50, t=80),
    scene=dict(
        xaxis=dict(title="Vencimiento (Años)", gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(title="Fecha", gridcolor='rgba(255,255,255,0.08)'),
        zaxis=dict(title="Rendimiento (%)", gridcolor='rgba(255,255,255,0.08)'),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    ),
)
plot3_path = os.path.join(PLOTS_DIR, 'chart3.html')
fig3.write_html(plot3_path, include_plotlyjs='cdn')
print(f"[OK] Grafico 3 guardado en: {plot3_path}")
print("\n[OK] Generacion de graficos completada exitosamente!")
