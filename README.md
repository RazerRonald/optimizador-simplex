# Optimizador Simplex 📊

Aplicación educativa en español para resolver programación lineal con Streamlit y el método simplex dual de HiGHS (SciPy).

## Funcionalidades

- Maximización y minimización con variables continuas no negativas.
- Tablas dinámicas de coeficientes y restricciones `<=`, `>=` y `=`.
- Valor óptimo, valores de variables y análisis de holguras y excedentes.
- Detección de modelos infactibles, no acotados y entradas inválidas.

## Ejecutar localmente

Usa Python 3.12 e instala las dependencias:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Inicia sesión en https://share.streamlit.io/ con tu cuenta de GitHub.
2. Crea una app desde el repositorio `RazerRonald/optimizador-simplex`.
3. Selecciona la rama `main` y el archivo principal `app.py`.
4. En las opciones avanzadas selecciona Python 3.12 y despliega.

Las dependencias se instalan desde `requirements.txt`. No necesita claves ni secretos.

## Ejemplo inicial

Maximizar `3X1 + 2X2`, sujeto a `X1 + X2 <= 4`, `X1 <= 2`, `X2 <= 3` y variables no negativas.
Resultado: `X1 = 2`, `X2 = 2`, `Z = 10`.

El programa muestra la solución final, no los tableaux de cada iteración del simplex.
