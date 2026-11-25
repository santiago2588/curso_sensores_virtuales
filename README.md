# Sensores Virtuales y Analizadores en Línea — Pungo Academy

Repositorio oficial del curso **Sensores Virtuales: Predicción en Tiempo Real de Variables de Calidad en Procesos Industriales** (**Pungo Tech**).

## Contenido del Repositorio
- `main.py`: Aplicación de Sala de Control en Streamlit con predicción puntual LightGBM, intervalos cuantílicos calibrados CQR (P10–P90), explicabilidad SHAP local en vivo, monitoreo multivariado PCA (Hotelling $T^2$ y SPE), corrección recursiva por Filtro de Kalman y simulador What-If.
- `sensor_virtual_v1.pkl`: Paquete de producción serializado generado desde el Notebook 07.
- `notebooks/`: Los 9 notebooks ejecutables de Google Colab (`00_Configuracion.ipynb` a `08_Caso_Estudio_Kamyr.ipynb`).
- `data/`: Dataset principal (`curso_sensor_raw_data.csv` — 1,095 turnos de 8h en unidad de destilación atmosférica).

## Ejecución Local
```bash
pip install -r requirements.txt
streamlit run main.py
```

---
**Pungo Tech** (`PUNGO TECH S.A.S.`) · [www.pungoapp.com](https://www.pungoapp.com) · contacto@pungoapp.com · Quito, Ecuador
