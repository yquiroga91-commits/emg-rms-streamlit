import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Modelo RMS sEMG",
    page_icon="💪",
    layout="wide"
)


# =========================
# Funciones del modelo
# =========================

def crear_entrada(t, A, tipo, inicio, fin, frecuencia):
    if tipo == "Escalón":
        return A * np.ones_like(t)

    if tipo == "Pulso":
        return A * ((t >= inicio) & (t <= fin)).astype(float)

    if tipo == "Senoidal":
        return A * 0.5 * (1 + np.sin(2 * np.pi * frecuencia * t))

    if tipo == "Rampa":
        if t[-1] == 0:
            return np.zeros_like(t)
        return A * t / t[-1]

    return np.zeros_like(t)


def simular_modelo(t, u, K, tau, y0, ruido=0.0):
    if tau <= 0:
        raise ValueError("La constante de tiempo τ debe ser mayor que cero.")

    y = np.zeros_like(t, dtype=float)
    y[0] = y0

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dydt = (-y[i - 1] + K * u[i - 1]) / tau
        y[i] = y[i - 1] + dt * dydt

    y_sin_ruido = y.copy()

    if ruido > 0:
        y = y + np.random.normal(0, ruido, size=len(y))

    return y, y_sin_ruido


def calcular_metricas(y_real, y_modelo):
    mse = np.mean((y_real - y_modelo) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_real - y_modelo) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return mse, rmse, r2


def respuesta_base_por_y0(t, tau, y0):
    u_cero = np.zeros_like(t)
    y_base, _ = simular_modelo(t, u_cero, K=0.0, tau=tau, y0=y0, ruido=0)
    return y_base


def mejor_K_para_tau_pulso(t, y_real, tau, y0, A, inicio, fin):
    u = crear_entrada(t, A, "Pulso", inicio, fin, frecuencia=0.0)

    y_base = respuesta_base_por_y0(t, tau, y0)
    y_unitaria, _ = simular_modelo(t, u, K=1.0, tau=tau, y0=0.0, ruido=0)

    objetivo = y_real - y_base
    denominador = np.dot(y_unitaria, y_unitaria)

    if denominador <= 1e-12:
        K_estimado = 0.001
    else:
        K_estimado = np.dot(y_unitaria, objetivo) / denominador
        K_estimado = max(0.001, K_estimado)

    y_modelo = y_base + K_estimado * y_unitaria
    mse, rmse, r2 = calcular_metricas(y_real, y_modelo)

    return K_estimado, y_modelo, mse, rmse, r2


def ajustar_modelo_pulso(t, y_real, A):
    if A == 0:
        raise ValueError("La amplitud de entrada A no puede ser cero.")

    y0_ajustado = float(y_real[0])
    t_min = float(t[0])
    t_max = float(t[-1])
    duracion = t_max - t_min

    if duracion <= 0:
        raise ValueError("El vector de tiempo no es válido.")

    t_pico = float(t[np.argmax(y_real)])

    mejor = {
        "rmse": np.inf,
        "K": None,
        "tau": None,
        "inicio": None,
        "fin": None,
        "y_modelo": None,
        "mse": None,
        "r2": None,
        "y0": y0_ajustado,
    }

    # Búsqueda gruesa. Ajustada para que corra rápido en Streamlit Cloud.
    inicio_max = min(t_max - 0.05 * duracion, max(t_min + 0.05 * duracion, t_pico))
    valores_inicio = np.linspace(t_min, inicio_max, 20)
    valores_fin = np.linspace(max(t_min + 0.05 * duracion, t_pico - 0.35 * duracion), t_max, 26)
    valores_tau = np.linspace(0.02, max(0.08, min(2.5, duracion)), 38)

    total = len(valores_inicio) * len(valores_fin) * len(valores_tau)
    progreso = st.progress(0, text="Ajustando modelo...")
    contador = 0

    for tau_prueba in valores_tau:
        for inicio_prueba in valores_inicio:
            for fin_prueba in valores_fin:
                contador += 1
                if contador % 300 == 0:
                    progreso.progress(min(contador / total, 1.0), text="Ajustando modelo...")

                if fin_prueba <= inicio_prueba:
                    continue

                K, y_modelo, mse, rmse, r2 = mejor_K_para_tau_pulso(
                    t, y_real, tau_prueba, y0_ajustado, A, inicio_prueba, fin_prueba
                )

                if rmse < mejor["rmse"]:
                    mejor.update({
                        "rmse": rmse,
                        "K": K,
                        "tau": tau_prueba,
                        "inicio": inicio_prueba,
                        "fin": fin_prueba,
                        "y_modelo": y_modelo,
                        "mse": mse,
                        "r2": r2,
                    })

    # Búsqueda fina alrededor del mejor resultado.
    ancho_inicio = 0.10 * duracion
    ancho_fin = 0.12 * duracion
    ancho_tau = 0.35 * mejor["tau"]

    valores_inicio_fino = np.linspace(
        max(t_min, mejor["inicio"] - ancho_inicio),
        min(t_max, mejor["inicio"] + ancho_inicio),
        15
    )
    valores_fin_fino = np.linspace(
        max(t_min, mejor["fin"] - ancho_fin),
        min(t_max, mejor["fin"] + ancho_fin),
        15
    )
    valores_tau_fino = np.linspace(
        max(0.005, mejor["tau"] - ancho_tau),
        max(0.006, mejor["tau"] + ancho_tau),
        22
    )

    total_fino = len(valores_inicio_fino) * len(valores_fin_fino) * len(valores_tau_fino)
    contador = 0

    for tau_prueba in valores_tau_fino:
        for inicio_prueba in valores_inicio_fino:
            for fin_prueba in valores_fin_fino:
                contador += 1
                if contador % 150 == 0:
                    progreso.progress(min(contador / total_fino, 1.0), text="Afinando ajuste...")

                if fin_prueba <= inicio_prueba:
                    continue

                K, y_modelo, mse, rmse, r2 = mejor_K_para_tau_pulso(
                    t, y_real, tau_prueba, y0_ajustado, A, inicio_prueba, fin_prueba
                )

                if rmse < mejor["rmse"]:
                    mejor.update({
                        "rmse": rmse,
                        "K": K,
                        "tau": tau_prueba,
                        "inicio": inicio_prueba,
                        "fin": fin_prueba,
                        "y_modelo": y_modelo,
                        "mse": mse,
                        "r2": r2,
                    })

    progreso.empty()
    return mejor


def leer_csv_validacion(archivo):
    df = pd.read_csv(archivo)
    df.columns = [c.strip() for c in df.columns]

    if "tiempo" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada 'tiempo'.")

    if "rms" in df.columns:
        col_rms = "rms"
    elif "rms_real" in df.columns:
        col_rms = "rms_real"
    else:
        raise ValueError("El CSV debe tener una columna llamada 'rms' o 'rms_real'.")

    df = df[["tiempo", col_rms]].copy()
    df = df.rename(columns={col_rms: "rms"})
    df = df.dropna()
    df = df.sort_values("tiempo")

    t = df["tiempo"].to_numpy(dtype=float)
    rms = df["rms"].to_numpy(dtype=float)

    if len(t) < 3:
        raise ValueError("El CSV no tiene suficientes datos válidos.")

    t = t - t[0]
    return t, rms


def figura_simulacion(t, u, y, y_sin_ruido):
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(t, u, label="Entrada u(t)")
    axs[0].set_title("Entrada de activación muscular")
    axs[0].set_ylabel("u(t)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, y_sin_ruido, label="RMS modelada sin ruido")
    axs[1].plot(t, y, label="RMS modelada con ruido", alpha=0.7)
    axs[1].set_title("Respuesta dinámica de la amplitud RMS")
    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("RMS")
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    return fig


def figura_validacion(t, y_real, y_modelo, u, A):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, y_real, label="RMS real")
    ax.plot(t, y_modelo, label="RMS modelada")

    escala = np.max(y_real) / max(abs(A), 1e-9)
    ax.plot(t, u * escala, "--", alpha=0.45, label="Pulso ajustado escalado")

    ax.set_title("Validación: RMS real vs RMS modelada")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("RMS")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def convertir_resultado_a_csv(t, y_real, y_modelo, u):
    df_export = pd.DataFrame({
        "tiempo": t,
        "rms_real": y_real,
        "rms_modelada": y_modelo,
        "entrada_u": u,
    })
    return df_export.to_csv(index=False).encode("utf-8")


# =========================
# Interfaz
# =========================

st.title("Modelo dinámico de amplitud RMS de sEMG")
st.caption("Versión web con Streamlit: simulación, validación y ajuste automático.")

with st.sidebar:
    st.header("Parámetros del sistema")

    K = st.number_input("Ganancia K", value=1.0, min_value=0.001, step=0.1, format="%.4f")
    tau = st.number_input("Constante de tiempo τ [s]", value=0.8, min_value=0.001, step=0.1, format="%.4f")
    y0 = st.number_input("Condición inicial y(0)", value=0.0, step=0.01, format="%.4f")
    A = st.number_input("Amplitud entrada A", value=1.0, min_value=0.001, step=0.1, format="%.4f")
    t_total = st.number_input("Tiempo total [s]", value=4.0, min_value=0.001, step=0.5, format="%.4f")
    dt = st.number_input("Paso dt [s]", value=0.01, min_value=0.0001, step=0.001, format="%.6f")
    ruido = st.number_input("Ruido RMS σ", value=0.0, min_value=0.0, step=0.01, format="%.4f")
    inicio_pulso = st.number_input("Inicio pulso [s]", value=0.9, min_value=0.0, step=0.05, format="%.4f")
    fin_pulso = st.number_input("Fin pulso [s]", value=1.65, min_value=0.0, step=0.05, format="%.4f")
    frecuencia = st.number_input("Frecuencia senoidal [Hz]", value=0.5, min_value=0.0, step=0.1, format="%.4f")
    tipo_entrada = st.selectbox("Tipo de entrada", ["Pulso", "Escalón", "Senoidal", "Rampa"])


tab_sim, tab_val, tab_modelo = st.tabs(["Simulación", "Validación", "Modelo matemático"])


with tab_sim:
    st.subheader("Simulación")

    if fin_pulso <= inicio_pulso and tipo_entrada == "Pulso":
        st.error("El fin del pulso debe ser mayor que el inicio del pulso.")
    else:
        t = np.arange(0, t_total + dt, dt)
        u = crear_entrada(t, A, tipo_entrada, inicio_pulso, fin_pulso, frecuencia)
        y, y_sin_ruido = simular_modelo(t, u, K, tau, y0, ruido)

        st.pyplot(figura_simulacion(t, u, y, y_sin_ruido))

        polo = -1 / tau
        c1, c2, c3 = st.columns(3)
        c1.metric("Polo del sistema", f"{polo:.4f}")
        c2.metric("Valor final aprox.", f"{y_sin_ruido[-1]:.4f}")
        c3.metric("Muestras", f"{len(t)}")

        df_sim = pd.DataFrame({
            "tiempo": t,
            "entrada_u": u,
            "rms_modelada": y,
            "rms_sin_ruido": y_sin_ruido,
        })

        st.download_button(
            "Descargar simulación CSV",
            df_sim.to_csv(index=False).encode("utf-8"),
            file_name="simulacion_emg.csv",
            mime="text/csv"
        )


with tab_val:
    st.subheader("Validación con CSV")
    st.write("Sube un archivo CSV con columnas `tiempo,rms` o `tiempo,rms_real`.")

    archivo = st.file_uploader("Cargar CSV de validación", type=["csv"])

    if archivo is not None:
        try:
            t_val, rms_real = leer_csv_validacion(archivo)
            st.success(f"CSV cargado correctamente: {len(t_val)} muestras, duración {t_val[-1]:.4f} s")

            st.line_chart(pd.DataFrame({"RMS real": rms_real}, index=t_val))

            if st.button("Ajustar K, τ, inicio y fin automáticamente", type="primary"):
                resultado = ajustar_modelo_pulso(t_val, rms_real, A)

                u_ajustada = crear_entrada(
                    t_val,
                    A,
                    "Pulso",
                    resultado["inicio"],
                    resultado["fin"],
                    frecuencia
                )

                st.pyplot(figura_validacion(t_val, rms_real, resultado["y_modelo"], u_ajustada, A))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("K ajustado", f"{resultado['K']:.4f}")
                c2.metric("τ ajustado [s]", f"{resultado['tau']:.4f}")
                c3.metric("Inicio pulso [s]", f"{resultado['inicio']:.4f}")
                c4.metric("Fin pulso [s]", f"{resultado['fin']:.4f}")

                c5, c6, c7 = st.columns(3)
                c5.metric("MSE", f"{resultado['mse']:.6f}")
                c6.metric("RMSE", f"{resultado['rmse']:.6f}")
                c7.metric("R²", f"{resultado['r2']:.4f}")

                csv_resultado = convertir_resultado_a_csv(
                    t_val,
                    rms_real,
                    resultado["y_modelo"],
                    u_ajustada
                )

                st.download_button(
                    "Descargar validación CSV",
                    csv_resultado,
                    file_name="export_validacion.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error al procesar el CSV: {e}")
    else:
        st.info("Primero sube el CSV de validación.")


with tab_modelo:
    st.subheader("Modelo matemático")
    st.markdown(
        r"""
El sistema representa la evolución temporal de la amplitud RMS de una señal electromiográfica de superficie durante una contracción muscular.

### Ecuación diferencial

$$
\tau \frac{dy(t)}{dt} + y(t) = K u(t)
$$

### Forma estándar

$$
\frac{dy(t)}{dt} = -\frac{1}{\tau}y(t) + \frac{K}{\tau}u(t)
$$

### Función de transferencia

$$
G(s) = \frac{Y(s)}{U(s)} = \frac{K}{\tau s + 1}
$$

Para una contracción tipo **closed fist**, la entrada puede aproximarse como un pulso:

$$
u(t) = A, \quad inicio \leq t \leq fin
$$

$$
u(t) = 0, \quad \text{en otro caso}
$$

El ajuste automático estima **K**, **τ**, **inicio del pulso** y **fin del pulso**.
"""
    )
