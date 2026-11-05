"""Ejecutar: streamlit run app.py
Instalar dependencias: python -m pip install streamlit scipy numpy pandas
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog


def resolver(objetivo, restricciones, sentido):
    """Valida los datos y convierte el modelo al formato de SciPy."""
    variables = list(objetivo.columns)
    try:
        c = objetivo.iloc[0].to_numpy(dtype=float)
        a = restricciones[variables].to_numpy(dtype=float)
        b = restricciones['RHS'].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Completa todos los coeficientes con números válidos.') from exc
    if not all(np.isfinite(v).all() for v in (c, a, b)):
        raise ValueError('No dejes celdas vacías ni uses valores infinitos.')
    signos = restricciones['Tipo'].to_numpy()
    if not np.isin(signos, ['<=', '>=', '=']).all():
        raise ValueError('Selecciona <=, >= o = en cada restricción.')

    desigualdad = signos != '='
    igualdad = signos == '='
    # Una restricción >= se transforma multiplicando AMBOS lados por -1.
    factores = np.where(signos[desigualdad] == '>=', -1.0, 1.0)
    maximizar = sentido == 'Maximizar'
    resultado = linprog(
        -c if maximizar else c,
        A_ub=a[desigualdad] * factores[:, None] if desigualdad.any() else None,
        b_ub=b[desigualdad] * factores if desigualdad.any() else None,
        A_eq=a[igualdad] if igualdad.any() else None,
        b_eq=b[igualdad] if igualdad.any() else None,
        bounds=[(0, None)] * len(c),
        method='highs-ds',  # Simplex dual revisado de HiGHS.
    )
    if not resultado.success:
        return resultado, None, None, None

    z = -resultado.fun if maximizar else resultado.fun
    resumen = pd.DataFrame({'Variable': variables, 'Valor óptimo': resultado.x,
                            'Coeficiente objetivo': c, 'Aporte a Z': c * resultado.x})
    lhs = a @ resultado.x
    # Calcular holguras con las restricciones ORIGINALES, antes del cambio de signo.
    filas = []
    for i, signo in enumerate(signos):
        margen = lhs[i] - b[i] if signo == '>=' else b[i] - lhs[i]
        tolerancia = 1e-7 * max(1.0, abs(lhs[i]), abs(b[i]))
        activa = abs(margen) <= tolerancia
        cumple = activa if signo == '=' else margen >= -tolerancia
        filas.append({
            'Restricción': f'R{i + 1}', 'Lado izquierdo': lhs[i], 'Tipo': signo,
            'RHS': b[i], 'Holgura / excedente / residual': margen,
            'Medida': {'<=': 'Holgura (RHS − LHS)', '>=': 'Excedente (LHS − RHS)',
                       '=': 'Residual (RHS − LHS)'}[signo],
            'Estado': 'Revisar tolerancia' if not cumple else (
                'Activa' if activa else 'Con margen'),
        })
    return resultado, z, resumen, pd.DataFrame(filas)


def main():
    st.set_page_config(page_title='Optimizador Simplex', page_icon='📊', layout='wide')
    st.markdown('''<style>
        .stButton button, .stFormSubmitButton button {border-radius: 10px;}
        [data-testid="stMetric"] {border: 1px solid #93a4b8; padding: 18px;
                                  border-radius: 12px;}
        </style>''', unsafe_allow_html=True)
    st.title('Optimizador Simplex 📊')
    st.write('Define tu modelo de programación lineal y encuentra su solución óptima.')
    st.caption('Variables continuas y no negativas: Xi ≥ 0. Motor: HiGHS, simplex dual.')

    # Las dimensiones están fuera del formulario para regenerar las tablas inmediatamente.
    with st.sidebar:
        st.header('Dimensiones del modelo')
        n = int(st.number_input('Variables de decisión', min_value=1, max_value=50, value=2))
        m = int(st.number_input('Restricciones', min_value=1, max_value=100, value=3))
        st.info('Cambiar las dimensiones reinicia las tablas. Usa punto para los decimales.')
        st.caption('Ejemplo inicial: maximizar 3X1 + 2X2, con X1 + X2 ≤ 4, '
                   'X1 ≤ 2 y X2 ≤ 3. Óptimo: Z = 10.')

    variables = [f'X{i + 1}' for i in range(n)]
    objetivo = pd.DataFrame(np.zeros((1, n)), columns=variables)
    restricciones = pd.DataFrame(np.zeros((m, n)), columns=variables)
    restricciones['Tipo'] = '<='
    restricciones['RHS'] = 0.0
    if (n, m) == (2, 3):
        objetivo.iloc[0] = [3.0, 2.0]
        restricciones[variables] = np.array([[1., 1.], [1., 0.], [0., 1.]])
        restricciones['RHS'] = [4.0, 2.0, 3.0]
    config = {v: st.column_config.NumberColumn(v, required=True) for v in variables}
    with st.form(f'modelo_{n}_{m}'):
        izquierda, derecha = st.columns([1, 2])
        with izquierda:
            sentido = st.radio('Objetivo', ['Maximizar', 'Minimizar'])
        with derecha:
            st.write('Introduce un coeficiente por variable; utiliza 0 si no participa.')
        st.subheader('Función objetivo')
        objetivo = st.data_editor(objetivo, column_config=config, hide_index=True,
                                 num_rows='fixed', key=f'objetivo_{n}_{m}')
        st.subheader('Restricciones')
        restricciones.index = [f'R{i + 1}' for i in range(m)]
        restricciones = st.data_editor(
            restricciones, num_rows='fixed', key=f'restricciones_{n}_{m}',
            column_config={**config,
                           'Tipo': st.column_config.SelectboxColumn(
                               'Tipo', options=['<=', '>=', '='], required=True),
                           'RHS': st.column_config.NumberColumn('RHS', required=True)})
        enviado = st.form_submit_button('Resolver modelo', type='primary')

    if not enviado:
        st.info('Completa las tablas y pulsa «Resolver modelo».')
        return
    try:
        resultado, z, resumen, holguras = resolver(objetivo, restricciones, sentido)
    except (ValueError, TypeError, OverflowError) as exc:
        st.error(f'Revisa los datos: {exc}')
        return
    if resultado.status == 2:
        st.error('Problema infactible: las restricciones se contradicen entre sí o con '
                 'la condición Xi ≥ 0. No existe una solución que las cumpla todas.')
    elif resultado.status == 3:
        st.warning('Problema no acotado: Z puede mejorar indefinidamente. '
                   'Revisa si faltan restricciones que limiten las variables.')
    elif not resultado.success:
        st.error('El solver no pudo confirmar una solución óptima. '
                 'Revisa la escala de los coeficientes y las restricciones.')
        st.code(resultado.message, language=None)
    else:
        st.success('¡Solución óptima encontrada!')
        izquierda, derecha = st.columns([1, 2])
        with izquierda:
            st.metric('Z máximo' if sentido == 'Maximizar' else 'Z mínimo', f'{z:.10g}')
        with derecha:
            st.subheader('Valores de las variables')
            st.dataframe(resumen, hide_index=True)
        st.subheader('Análisis de holguras y excedentes')
        st.dataframe(holguras, hide_index=True)
        st.caption('En ≤, la holgura indica cuánto queda disponible. En ≥, el excedente '
                   'indica cuánto se supera el mínimo exigido. En =, el residual debe '
                   'ser aproximadamente cero. Una restricción activa no tiene margen. '
                   'Los valores cercanos a cero pueden reflejar redondeo numérico.')
        st.caption('Resultados correspondientes al último envío del formulario. '
                   'Tras editar una celda, pulsa Resolver modelo para recalcular.')


if __name__ == '__main__':
    main()
