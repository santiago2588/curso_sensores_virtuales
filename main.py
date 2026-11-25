"""
main.py — Sensor Virtual: Flash Point del Keroseno (Control Room HMI v2.0)
Aplicacion Streamlit con Intervalos Cuantilicos (P10-P90), Explicabilidad SHAP,
Monitoreo de Dominio (Hotelling T2 / SPE) y Correccion de Sesgo por Filtro de Kalman.

Pungo Tech · www.pungoapp.com · contacto@pungoapp.com · Quito, Ecuador
Uso:
    streamlit run main.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

st.set_page_config(
    page_title="Sensor Virtual · Flash Point Keroseno | Pungo Tech",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo visual Pungo Tech 2027
st.markdown(
    """
    <style>
    .pungo-header {
        background: linear-gradient(135deg, #0D202B 0%, #004E6B 100%);
        color: #FFFFFF;
        padding: 1.1rem 1.5rem;
        border-radius: 12px;
        border-bottom: 4px solid #8EC14B;
        margin-bottom: 1rem;
    }
    .pungo-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; color: #FFFFFF; }
    .pungo-header p { margin: 0.25rem 0 0 0; font-size: 0.92rem; color: #CFE8F3; }
    .status-card {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 0.5rem;
    }
    .status-ok   { background: #ECFDF5; border-color: #27AE60; color: #065F46; }
    .status-warn { background: #FFFBEB; border-color: #E8820C; color: #92400E; }
    .status-bad  { background: #FEF2F2; border-color: #C0392B; color: #991B1B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pungo-header">
        <h1>Sensor Virtual en Tiempo Real: Flash Point del Keroseno</h1>
        <p>Unidad de Destilación Atmosférica · Analizador Inferencial v2.0 (LightGBM + CQR + SHAP + T²/SPE + Kalman) | <b>Pungo Tech</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    for candidate in ["sensor_virtual_v1.pkl", "/content/sensor_virtual_v1.pkl", "app/sensor_virtual_v1.pkl"]:
        if os.path.exists(candidate):
            return joblib.load(candidate)
    raise FileNotFoundError("No se encontró sensor_virtual_v1.pkl en el directorio actual ni en /content/.")


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


PRESETS = {
    "Operación Normal (Crudo API 33.5)": {
        "kerosene_draw_temp_C": 179.5,
        "column_top_temp_C": 109.5,
        "tray_15_temp_C": 200.0,
        "stripping_steam_flow_kgh": 2190.0,
        "kerosene_product_flow_m3h": 109.0,
        "reflux_ratio": 2.08,
        "column_delta_p_kPa": 20.0,
        "column_top_pressure_kPag": 110.0,
        "ambient_temp_C": 15.0,
        "crude_api_gravity": 33.5,
        "feed_flow_rate_m3h": 955.0,
        "feed_temperature_C": 364.5,
        "column_bottom_temp_C": 355.0,
        "reflux_flow_m3h": 298.0,
        "condenser_outlet_temp_C": 44.5,
        "kalman_bias": 0.0,
    },
    "Alerta Límite Bajo (< 40 °C)": {
        "kerosene_draw_temp_C": 171.8,
        "column_top_temp_C": 111.5,
        "tray_15_temp_C": 192.5,
        "stripping_steam_flow_kgh": 1880.0,
        "kerosene_product_flow_m3h": 112.0,
        "reflux_ratio": 1.96,
        "column_delta_p_kPa": 21.8,
        "column_top_pressure_kPag": 114.5,
        "ambient_temp_C": 4.0,
        "crude_api_gravity": 33.2,
        "feed_flow_rate_m3h": 970.0,
        "feed_temperature_C": 361.0,
        "column_bottom_temp_C": 350.0,
        "reflux_flow_m3h": 286.0,
        "condenser_outlet_temp_C": 41.2,
        "kalman_bias": 0.0,
    },
    "Cambio de Crudo (API 29.0)": {
        "kerosene_draw_temp_C": 181.2,
        "column_top_temp_C": 110.8,
        "tray_15_temp_C": 203.2,
        "stripping_steam_flow_kgh": 2290.0,
        "kerosene_product_flow_m3h": 105.5,
        "reflux_ratio": 2.14,
        "column_delta_p_kPa": 20.6,
        "column_top_pressure_kPag": 111.0,
        "ambient_temp_C": 22.0,
        "crude_api_gravity": 29.0,
        "feed_flow_rate_m3h": 948.0,
        "feed_temperature_C": 367.0,
        "column_bottom_temp_C": 360.5,
        "reflux_flow_m3h": 308.0,
        "condenser_outlet_temp_C": 46.8,
        "kalman_bias": -0.6,
    },
    "Crudo Extra-Pesado + Deriva (T² Alto)": {
        "kerosene_draw_temp_C": 188.5,
        "column_top_temp_C": 112.6,
        "tray_15_temp_C": 206.8,
        "stripping_steam_flow_kgh": 2480.0,
        "kerosene_product_flow_m3h": 99.0,
        "reflux_ratio": 2.25,
        "column_delta_p_kPa": 22.4,
        "column_top_pressure_kPag": 115.5,
        "ambient_temp_C": 28.0,
        "crude_api_gravity": 26.2,
        "feed_flow_rate_m3h": 932.0,
        "feed_temperature_C": 370.5,
        "column_bottom_temp_C": 366.0,
        "reflux_flow_m3h": 324.0,
        "condenser_outlet_temp_C": 49.5,
        "kalman_bias": -2.8,
    },
}


def init_state():
    defaults = PRESETS["Operación Normal (Crudo API 33.5)"]
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def build_feature_row(vals: dict, feature_cols: list[str]) -> pd.DataFrame:
    temp_grad = vals["kerosene_draw_temp_C"] - vals["column_top_temp_C"]
    tray_diff = vals["tray_15_temp_C"] - vals["kerosene_draw_temp_C"]
    spec_steam = vals["stripping_steam_flow_kgh"] / (vals["kerosene_product_flow_m3h"] + 1e-6)
    yield_ratio = vals["kerosene_product_flow_m3h"] / (vals["feed_flow_rate_m3h"] + 1e-6)
    cond_approach = vals["condenser_outlet_temp_C"] - vals["ambient_temp_C"]

    row = {
        **vals,
        "temp_gradient": temp_grad,
        "tray_to_draw_diff": tray_diff,
        "specific_steam": spec_steam,
        "yield_ratio": yield_ratio,
        "condenser_approach": cond_approach,
        "kerosene_draw_temp_lag1": vals["kerosene_draw_temp_C"] - 0.3,
        "kerosene_draw_temp_lag2": vals["kerosene_draw_temp_C"] - 0.6,
        "kerosene_draw_temp_ma9": vals["kerosene_draw_temp_C"],
        "temp_gradient_lag1": temp_grad - 0.2,
        "post_crude_change": 1 if vals["crude_api_gravity"] < 31.0 else 0,
    }
    return pd.DataFrame([[row.get(c, 0.0) for c in feature_cols]], columns=feature_cols)


def predict_full(bundle: dict, vals: dict, kalman_bias: float):
    X_row = build_feature_row(vals, bundle["feature_cols"])
    pred_raw = float(bundle["model_point"].predict(X_row)[0])
    pred = float(np.clip(pred_raw + kalman_bias, 30.0, 72.0))

    cqr = float(bundle.get("cqr_offset", 0.0))
    q10 = float(np.clip(bundle["model_q10"].predict(X_row)[0] - cqr + kalman_bias, 28.0, pred))
    q90 = float(np.clip(bundle["model_q90"].predict(X_row)[0] + cqr + kalman_bias, pred, 74.0))

    # PCA Hotelling T2 & SPE domain monitoring
    pca_mon = bundle["pca_monitor"]
    X_s = pca_mon["scaler"].transform(X_row)
    scores = pca_mon["pca"].transform(X_s)
    t2_val = float(np.sum((scores ** 2) / pca_mon["eigenvalues"], axis=1)[0])
    rec = pca_mon["pca"].inverse_transform(scores)
    spe_val = float(np.sum((X_s - rec) ** 2, axis=1)[0])

    # SHAP explanation
    explainer = get_explainer(bundle["model_point"])
    sv = explainer.shap_values(X_row)[0]
    base_val = float(np.atleast_1d(explainer.expected_value)[0])
    shap_series = pd.Series(sv, index=bundle["feature_cols"])

    return {
        "X_row": X_row,
        "pred": pred,
        "pred_raw": pred_raw,
        "q10": q10,
        "q90": q90,
        "t2": t2_val,
        "t2_lim95": float(pca_mon["t2_limit_95"]),
        "t2_lim99": float(pca_mon["t2_limit_99"]),
        "spe": spe_val,
        "spe_lim95": float(pca_mon["spe_limit_95"]),
        "spe_lim99": float(pca_mon["spe_limit_99"]),
        "shap": shap_series,
        "shap_base": base_val,
    }


try:
    bundle = load_bundle()
    init_state()
    bundle_ok = True
except Exception as exc:
    st.error(f"Error cargando modelo: {exc}")
    bundle_ok = False

if bundle_ok:
    st.sidebar.header("Simulador de Sala de Control")
    st.sidebar.markdown("**Escenarios Operativos Rápidos**")
    preset_choice = st.sidebar.selectbox(
        "Cargar escenario predefinido",
        list(PRESETS.keys()),
        index=0,
    )
    if st.sidebar.button("Aplicar Escenario Seleccionado", use_container_width=True):
        apply_preset(preset_choice)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Variables Clave del DCS")
    st.sidebar.slider("Temp. extracción keroseno (°C)", 165.0, 205.0, step=0.2, key="kerosene_draw_temp_C")
    st.sidebar.slider("Temp. tope de torre (°C)", 103.0, 118.0, step=0.2, key="column_top_temp_C")
    st.sidebar.slider("Temp. Plato 15 (°C)", 182.0, 222.0, step=0.5, key="tray_15_temp_C")
    st.sidebar.slider("Vapor de despojamiento (kg/h)", 1650.0, 2750.0, step=10.0, key="stripping_steam_flow_kgh")
    st.sidebar.slider("Flujo producto keroseno (m³/h)", 85.0, 135.0, step=0.5, key="kerosene_product_flow_m3h")
    st.sidebar.slider("Relación de reflujo", 1.85, 2.35, step=0.01, key="reflux_ratio")
    st.sidebar.slider("Caída de presión ΔP (kPa)", 17.0, 24.5, step=0.1, key="column_delta_p_kPa")
    st.sidebar.slider("Presión tope torre (kPag)", 102.0, 122.0, step=0.5, key="column_top_pressure_kPag")
    st.sidebar.slider("Temperatura ambiente (°C)", -5.0, 35.0, step=0.5, key="ambient_temp_C")
    st.sidebar.slider("Gravedad API del crudo", 24.5, 36.0, step=0.1, key="crude_api_gravity")

    with st.sidebar.expander("Variables Secundarias del DCS"):
        st.slider("Caudal alimentación (m³/h)", 880.0, 1040.0, step=2.0, key="feed_flow_rate_m3h")
        st.slider("Temp. alimentación (°C)", 352.0, 378.0, step=0.5, key="feed_temperature_C")
        st.slider("Temp. fondo de torre (°C)", 342.0, 372.0, step=0.5, key="column_bottom_temp_C")
        st.slider("Flujo de reflujo (m³/h)", 260.0, 350.0, step=1.0, key="reflux_flow_m3h")
        st.slider("Temp. salida condensador (°C)", 35.0, 56.0, step=0.5, key="condenser_outlet_temp_C")

    st.sidebar.divider()
    st.sidebar.subheader("Filtro de Kalman (Lab Bias)")
    kalman_bias = st.sidebar.slider(
        "Corrección de sesgo Kalman (°C)",
        -4.5,
        4.5,
        step=0.1,
        key="kalman_bias",
        help="Offset recursivo estimado contra las últimas muestras de laboratorio cada 8 h.",
    )

    current_vals = {c: float(st.session_state[c]) for c in PRESETS["Operación Normal (Crudo API 33.5)"] if c != "kalman_bias"}
    out = predict_full(bundle, current_vals, kalman_bias)

    pred = out["pred"]
    q10, q90 = out["q10"], out["q90"]
    t2, t2_lim = out["t2"], out["t2_lim95"]
    spe, spe_lim = out["spe"], out["spe_lim95"]
    domain_ok = (t2 <= t2_lim) and (spe <= spe_lim)

    # Clasificación operativa de especificación (38 - 60 C)
    if pred < 38.0:
        spec_css, spec_label = "status-bad", "FUERA DE ESPECIFICACIÓN (BAJO < 38 °C)"
        action_msg = "Riesgo de seguridad: subir temperatura de extracción o vapor de despojamiento; retener envío a tanque."
    elif pred > 60.0:
        spec_css, spec_label = "status-bad", "FUERA DE ESPECIFICACIÓN (ALTO > 60 °C)"
        action_msg = "Fracción pesada excesiva: bajar temperatura de extracción o incrementar reflujo superior."
    elif pred < 40.5 or pred > 57.5:
        spec_css, spec_label = "status-warn", "ADVERTENCIA — CERCA DEL LÍMITE"
        action_msg = "Monitoreo estrecho: ajustar ligeramente punto de corte para recuperar margen hacia 48 °C."
    else:
        spec_css, spec_label = "status-ok", "DENTRO DE ESPECIFICACIÓN (38–60 °C)"
        action_msg = "Operación nominal estable. Mantener lazo de control actual."

    # Fila 1: KPI Cards
    k1, k2, k3, k4 = st.columns([1.2, 1.5, 1.3, 1.1])
    with k1:
        st.metric(
            "Flash Point Predicho",
            f"{pred:.1f} °C",
            delta=f"{pred - 48.0:+.1f} °C vs objetivo (48 °C)",
        )
        st.caption(f"Intervalo 80% (CQR): **[{q10:.1f} – {q90:.1f} °C]**")

    with k2:
        st.markdown(
            f'<div class="status-card {spec_css}"><b>{spec_label}</b><br><span style="font-size:0.86rem;">{action_msg}</span></div>',
            unsafe_allow_html=True,
        )

    with k3:
        if domain_ok:
            dom_css = "status-ok"
            dom_title = "DOMINIO VÁLIDO (T² y SPE OK)"
            dom_sub = f"T² = {t2:.1f} (lím {t2_lim:.1f}) · SPE = {spe:.2f} (lím {spe_lim:.2f})"
        else:
            dom_css = "status-warn" if (t2 <= out["t2_lim99"] and spe <= out["spe_lim99"]) else "status-bad"
            dom_title = "ALERTA DE DOMINIO (EXTRAPOLACIÓN / DERIVA)"
            dom_sub = f"T² = {t2:.1f} (lím {t2_lim:.1f}) · SPE = {spe:.2f} (lím {spe_lim:.2f})"
        st.markdown(
            f'<div class="status-card {dom_css}"><b>{dom_title}</b><br><span style="font-size:0.84rem;">{dom_sub}</span></div>',
            unsafe_allow_html=True,
        )

    with k4:
        marg_inf = pred - 38.0
        marg_sup = 60.0 - pred
        st.metric("Margen Inferior (38 °C)", f"{marg_inf:+.1f} °C")
        st.caption(f"Margen sup: **{marg_sup:+.1f} °C** · Bias Kalman: **{kalman_bias:+.1f} °C**")

    st.divider()

    # Fila 2: Gauge Operativo + Explicabilidad SHAP en Vivo
    col_gauge, col_shap = st.columns([1.05, 1.15])

    with col_gauge:
        st.subheader("Indicador Operativo y Banda de Incertidumbre (80%)")
        fig_g, ax_g = plt.subplots(figsize=(7.2, 3.2), dpi=150)
        # Bandas de especificación
        ax_g.axvspan(30, 38, color="#FADBD8", alpha=0.85, label="Fuera espec. (<38 °C)")
        ax_g.axvspan(38, 40.5, color="#FCF3CF", alpha=0.90, label="Zona advertencia")
        ax_g.axvspan(40.5, 57.5, color="#D5F5E3", alpha=0.85, label="Zona óptima (38–60 °C)")
        ax_g.axvspan(57.5, 60, color="#FCF3CF", alpha=0.90)
        ax_g.axvspan(60, 72, color="#FADBD8", alpha=0.85)
        ax_g.axvline(38, color="#C0392B", linestyle="--", linewidth=1.6)
        ax_g.axvline(48, color="#007297", linestyle=":", linewidth=1.8, label="Objetivo (48.0 °C)")
        ax_g.axvline(60, color="#C0392B", linestyle="--", linewidth=1.6)

        # Banda P10-P90 + marcador puntual
        ax_g.hlines(0.5, q10, q90, color="#0D202B", linewidth=10, alpha=0.35, label=f"IC 80% [{q10:.1f}–{q90:.1f} °C]")
        ax_g.scatter([pred], [0.5], color="#0D202B", s=260, zorder=5, edgecolor="#FFFFFF", linewidth=2)
        ax_g.text(
            pred, 0.70, f"{pred:.1f} °C", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#0D202B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#0D202B", linewidth=1.2, alpha=0.95),
            zorder=6,
        )
        ax_g.text(
            (q10 + q90) / 2, 0.25, f"IC 80%: {q10:.1f} – {q90:.1f} °C",
            ha="center", va="top", fontsize=9.5, color="#0D202B", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.92),
            zorder=6,
        )

        ax_g.set_xlim(30, 72)
        ax_g.set_ylim(0, 1)
        ax_g.set_yticks([])
        ax_g.set_xlabel("Flash Point (°C)", fontsize=10, fontweight="bold")
        ax_g.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.92)
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close(fig_g)

    with col_shap:
        st.subheader("Explicabilidad SHAP en Vivo (¿Qué impulsa la predicción?)")
        top_sv = out["shap"].reindex(out["shap"].abs().sort_values(ascending=False).head(6).index)
        fig_s, ax_s = plt.subplots(figsize=(7.2, 3.2), dpi=150)
        vals_s = top_sv.values[::-1]
        names_s = top_sv.index[::-1]
        bar_colors = ["#007297" if v >= 0 else "#E8820C" for v in vals_s]
        bars = ax_s.barh(names_s, vals_s, color=bar_colors, edgecolor="white", height=0.62)
        ax_s.axvline(0, color="#0D202B", linewidth=1.0)
        for b, val in zip(bars, vals_s):
            offset = 0.06 if val >= 0 else -0.06
            ha = "left" if val >= 0 else "right"
            ax_s.text(val + offset, b.get_y() + b.get_height() / 2, f"{val:+.2f} °C", va="center", ha=ha, fontsize=8.5, fontweight="bold")
        max_abs = max(1.2, float(np.max(np.abs(vals_s))) * 1.28)
        ax_s.set_xlim(-max_abs, max_abs)
        ax_s.set_xlabel("Contribución marginal SHAP (°C respecto al valor base)", fontsize=9.5)
        ax_s.set_title(f"Base = {out['shap_base']:.1f} °C → Predicción sin Kalman = {out['pred_raw']:.1f} °C", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    # Fila 3: Pestañas Analíticas
    tab1, tab2, tab3 = st.tabs([
        "Monitor de Deriva y Filtro de Kalman",
        "Dominio de Aplicabilidad (Hotelling T² y SPE)",
        "Simulador What-If (Sensibilidad)",
    ])

    hist_df = bundle.get("history_eval")
    with tab1:
        if hist_df is not None:
            fig_d, axes = plt.subplots(2, 1, figsize=(12, 5.6), dpi=150, sharex=True)
            x_dates = pd.to_datetime(hist_df["date"])
            axes[0].fill_between(x_dates, hist_df["ci_low"], hist_df["ci_high"], color="#007297", alpha=0.14, label="Banda CQR 80%")
            axes[0].plot(x_dates, hist_df["y_true"], color="#0D202B", linewidth=1.4, label="Laboratorio (Real)")
            axes[0].plot(x_dates, hist_df["y_pred_static"], color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.85, label="Modelo Estático (deriva por termopozo)")
            axes[0].plot(x_dates, hist_df["y_pred_kalman"], color="#8EC14B", linewidth=1.8, label="LightGBM + Filtro de Kalman")
            axes[0].axhline(38, color="#C0392B", linestyle=":", linewidth=1.0)
            axes[0].axhline(60, color="#C0392B", linestyle=":", linewidth=1.0)
            axes[0].set_ylabel("Flash Point (°C)")
            axes[0].set_title("Evaluación en Producción (Q4): Corrección de Deriva +4.2 °C mediante Filtro de Kalman", fontweight="bold")
            axes[0].legend(loc="upper left", fontsize=8.5, ncol=2)

            axes[1].plot(x_dates, hist_df["rolling_rmse_static"], color="#C0392B", linewidth=1.5, label="RMSE deslizante Estático (45 turnos)")
            axes[1].plot(x_dates, hist_df["rolling_rmse_kalman"], color="#007297", linewidth=1.8, label="RMSE deslizante + Kalman")
            axes[1].axhline(2.2, color="#E8820C", linestyle="--", linewidth=1.2, label="Umbral de alerta (2.2 °C)")
            axes[1].set_ylabel("RMSE (°C)")
            axes[1].set_xlabel("Fecha")
            axes[1].legend(loc="upper left", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)
        else:
            st.info("No hay histórico de evaluación adjunto en el archivo del modelo.")

    with tab2:
        if hist_df is not None:
            fig_t, axes = plt.subplots(2, 1, figsize=(12, 5.2), dpi=150, sharex=True)
            x_dates = pd.to_datetime(hist_df["date"])
            axes[0].plot(x_dates, hist_df["t2"], color="#007297", linewidth=1.4, label="Hotelling T²")
            axes[0].axhline(t2_lim, color="#E8820C", linestyle="--", linewidth=1.3, label=f"Límite 95% ({t2_lim:.1f})")
            axes[0].axhline(out["t2_lim99"], color="#C0392B", linestyle=":", linewidth=1.3, label=f"Límite 99% ({out['t2_lim99']:.1f})")
            axes[0].set_ylabel("T²")
            axes[0].set_title("Monitoreo Multivariado PCA: Detección Temprana de Cambio de Crudo y Deriva", fontweight="bold")
            axes[0].legend(loc="upper left", fontsize=8.5)

            axes[1].plot(x_dates, hist_df["spe"], color="#8EC14B", linewidth=1.4, label="SPE (Q-statistic)")
            axes[1].axhline(spe_lim, color="#E8820C", linestyle="--", linewidth=1.3, label=f"Límite 95% ({spe_lim:.2f})")
            axes[1].axhline(out["spe_lim99"], color="#C0392B", linestyle=":", linewidth=1.3, label=f"Límite 99% ({out['spe_lim99']:.2f})")
            axes[1].set_ylabel("SPE")
            axes[1].set_xlabel("Fecha")
            axes[1].legend(loc="upper left", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

    with tab3:
        st.markdown("Simula el impacto de ajustar la **Temperatura de Extracción del Keroseno** (`±8 °C`) manteniendo el resto de la torre constante:")
        sweep_temps = np.linspace(max(165.0, current_vals["kerosene_draw_temp_C"] - 8.0), min(205.0, current_vals["kerosene_draw_temp_C"] + 8.0), 35)
        preds_w, lo_w, hi_w = [], [], []
        cqr = float(bundle.get("cqr_offset", 0.0))
        for td in sweep_temps:
            v_copy = dict(current_vals)
            v_copy["kerosene_draw_temp_C"] = float(td)
            Xr = build_feature_row(v_copy, bundle["feature_cols"])
            p = float(np.clip(bundle["model_point"].predict(Xr)[0] + kalman_bias, 30.0, 72.0))
            preds_w.append(p)
            lo_w.append(float(np.clip(bundle["model_q10"].predict(Xr)[0] - cqr + kalman_bias, 28.0, p)))
            hi_w.append(float(np.clip(bundle["model_q90"].predict(Xr)[0] + cqr + kalman_bias, p, 74.0)))

        fig_w, ax_w = plt.subplots(figsize=(11, 3.8), dpi=150)
        ax_w.fill_between(sweep_temps, lo_w, hi_w, color="#007297", alpha=0.18, label="Intervalo CQR 80%")
        ax_w.plot(sweep_temps, preds_w, color="#0D202B", linewidth=2.2, label="Flash Point predicho")
        ax_w.scatter([current_vals["kerosene_draw_temp_C"]], [pred], color="#8EC14B", s=110, zorder=5, edgecolor="#0D202B", label="Punto actual")
        ax_w.axhline(38.0, color="#C0392B", linestyle="--", label="Límite inf (38 °C)")
        ax_w.axhline(48.0, color="#007297", linestyle=":", label="Objetivo (48 °C)")
        ax_w.axhline(60.0, color="#C0392B", linestyle="--", label="Límite sup (60 °C)")
        ax_w.set_xlabel("Temp. extracción keroseno (°C)")
        ax_w.set_ylabel("Flash Point predicho (°C)")
        ax_w.set_title("Curva What-If de Sensibilidad Operativa", fontweight="bold")
        ax_w.legend(loc="upper left", fontsize=8.5, ncol=3)
        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    st.divider()
    st.caption(
        f"Sensor Virtual v{bundle.get('version', '2.0.0')} | LightGBM + CQR + SHAP + Hotelling T²/SPE + Kalman | "
        "Pungo Tech · www.pungoapp.com · contacto@pungoapp.com · Quito, Ecuador"
    )
