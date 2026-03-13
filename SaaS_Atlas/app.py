import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import os
import io
import warnings
from datetime import datetime

# --- Global Configurations ---
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Ensure output directory exists (relative to app file)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UI Configuration (SaaS / Power BI Aesthetic) ---
st.set_page_config(
    page_title="SaaS Atlas | Professional Data Portal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💠"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stApp { background-color: #f0f2f5; }
    .main-header { color: #1f4e79; font-weight: 600; margin-bottom: 20px; }
    .card {
        background-color: white; padding: 25px; border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e1e4e8;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; background-color: #fff; border-radius: 4px 4px 0 0;
        border: 1px solid #e1e4e8; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #1f4e79 !important; color: white !important; }
    .sidebar-section { background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Math Core & Fitting ---
def sigmoid(t, a, b, t0, k):
    with np.errstate(over='ignore', invalid='ignore'):
        return a + (b / (1 + np.exp(-(t - t0) / k)))

def exponential(x, a, b):
    with np.errstate(over='ignore', invalid='ignore'):
        return a * (1 - np.exp(-x / b))

def perform_fit(x, y, fit_type, x_range=None):
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3: return None, None, {"Error": "Insufficient data"}
    
    sort_idx = np.argsort(x)
    x_s, y_s = x[sort_idx], y[sort_idx]
    
    if x_range is not None:
        x_p = np.linspace(x_range[0], x_range[1], 300)
    else:
        x_p = np.linspace(x.min(), x.max(), 300)
        
    y_f, params = None, {}
    ci_data = None
    
    try:
        if fit_type == "Sigmoid":
            p0 = [min(y_s), max(y_s)-min(y_s), np.median(x_s), 1.0]
            bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])
            popt, _ = curve_fit(sigmoid, x_s, y_s, p0=p0, bounds=bounds, maxfev=10000)
            y_f = sigmoid(x_p, *popt)
            a, b, t0, k = popt
            params = {
                'a': a, 'b': b, 't0': t0, 'k': k,
                'Initial Length (µm)': y_f[0],
                'Final Length (µm)': y_f[-1],
                'Rate (µm/min)': (b/(4*k))*60 if k!=0 else 0
            }
        elif fit_type == "Exponential":
            p0 = [max(y_s), 50.0]
            popt, _ = curve_fit(exponential, x_s, y_s, p0=p0, maxfev=10000)
            y_f = exponential(x_p, *popt)
            params = {
                'Amp (a)': popt[0], 'T (b)': popt[1], 
                'Initial Length (µm)': y_f[0],
                'Final Length (µm)': y_f[-1],
                'Speed (µm/min)': (popt[0]/popt[1])*60 if popt[1]!=0 else 0
            }
        elif fit_type == "Linear (Advanced)":
            fitresult, cov = np.polyfit(x_s, y_s, 1, cov=True)
            slope, intercept = fitresult
            slope_std_dev = np.sqrt(cov[0, 0])
            t_stat = slope / slope_std_dev
            df = len(x_s) - 2
            p_val = 2 * (1 - t.cdf(np.abs(t_stat), df))
            
            y_f = np.polyval(fitresult, x_p)
            
            # CI Calculation
            y_pred_s = np.polyval(fitresult, x_s)
            SE_y = np.sqrt(np.sum((y_s - y_pred_s)**2) / df)
            SSD_x = np.sum((x_s - np.mean(x_s))**2)
            SE_x_p = SE_y * np.sqrt(1/len(x_s) + (x_p - np.mean(x_s))**2 / SSD_x)
            ci_val = t.ppf(0.975, df) * SE_x_p
            upper_ci = y_f + ci_val
            lower_ci = y_f - ci_val
            ci_data = (upper_ci, lower_ci)
            
            r2 = r2_score(y_s, y_pred_s)
            params = {
                "Slope": slope, "Intercept": intercept, "P-Value": p_val, "R-Squared": r2,
                'Initial Length (µm)': y_f[0], 'Final Length (µm)': y_f[-1]
            }
        elif fit_type == "Linear":
            z = np.polyfit(x_s, y_s, 1); y_f = np.poly1d(z)(x_p)
            params = {
                "Slope": z[0], "Intercept": z[1],
                'Initial Length (µm)': y_f[0], 'Final Length (µm)': y_f[-1]
            }
        elif "ML:" in fit_type:
            model = RandomForestRegressor(n_estimators=50) if "Forest" in fit_type else make_pipeline(StandardScaler(), SVR())
            model.fit(x_s.reshape(-1,1), y_s)
            y_f = model.predict(x_p.reshape(-1,1))
            params = {
                "Model": fit_type, "Score": model.score(x_s.reshape(-1,1), y_s),
                'Initial Length (µm)': y_f[0], 'Final Length (µm)': y_f[-1]
            }
    except Exception as e: params = {"Error": str(e)}
    return x_p, y_f, params, ci_data

# --- Sidebar: Global Controls ---
st.sidebar.markdown("<h2 style='color:#1f4e79;'>💠 SaaS Atlas</h2>", unsafe_allow_html=True)

# --- About Section ---
with st.sidebar.expander("ℹ️ About SaaS Atlas", expanded=False):
    st.markdown("""
    **SaaS Atlas Platform**
    A professional-grade data analysis portal for biological and physical trend fitting.
    
    **Capabilities:**
    - Multi-tenant data management.
    - Automated non-linear curve fitting.
    - ML-based regression (SVR, Random Forest).
    - High-resolution spatial 3D visualization.
    
    **Mathematical Fit Functions:**
    1. **Sigmoid:** $f(t) = a + \\frac{b}{1 + e^{-(t-t_0)/k}}$. Used for growth curves and saturation processes.
    2. **Exponential:** $f(x) = a(1 - e^{-x/b})$. Used for rise-to-maximum dynamics.
    3. **Linear:** $y = mx + c$. Standard rate analysis.
    4. **ML Models:** Non-parametric fitting for complex non-linear trends where biological mechanisms are unknown.
    """)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Exit Platform", type="primary", use_container_width=True):
    st.sidebar.warning("Shutting down kernel...")
    os._exit(0)

st.sidebar.subheader("📂 Data Acquisition")
files = st.sidebar.file_uploader("Upload Batch (CSV, Excel)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
ignore_nan = st.sidebar.checkbox("Ignore NaN (Per Analysis)", value=True)

all_data = {}
if files:
    for f in files:
        try:
            if f.name.endswith('.csv'): 
                try:
                    df = pd.read_csv(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, encoding='cp1252')
                all_data[f.name] = df
            else:
                xl = pd.ExcelFile(f, engine='openpyxl')
                sheet_names = xl.sheet_names
                if len(sheet_names) > 1:
                    # Let the user select sheets for each Excel file
                    selected_sheets = st.sidebar.multiselect(f"Sheets in {f.name}", sheet_names, default=[sheet_names[0]], key=f"sheets_{f.name}")
                    for sheet in selected_sheets:
                        all_data[f"{f.name} ({sheet})"] = pd.read_excel(f, sheet_name=sheet, engine='openpyxl')
                else:
                    all_data[f.name] = pd.read_excel(f, engine='openpyxl')
        except Exception as e: st.sidebar.error(f"Failed {f.name}: {e}")

all_cols = list(set.union(*[set(df.columns) for df in all_data.values()])) if all_data else []

st.sidebar.subheader("⚙️ Visualization Global")
vx = st.sidebar.selectbox("Primary X-Axis", all_cols, index=0 if 'time' not in all_cols else all_cols.index('time'))
vy_sel = st.sidebar.multiselect("Active Y-Axis Columns", all_cols, default=all_cols[1:2] if len(all_cols)>1 else [])
files_to_plot = st.sidebar.multiselect("Datasets to Visualize", list(all_data.keys()), default=list(all_data.keys())[:3])
show_legend = st.sidebar.checkbox("Show Plot Legend", value=True)

# Color Customization
st.sidebar.subheader("🎨 Color Customization")
custom_colors = {}
use_custom_colors = False
if all_data and files_to_plot and vy_sel:
    use_custom_colors = st.sidebar.toggle("Use Custom Trace Colors", value=False)
    if use_custom_colors:
        for fn in files_to_plot:
            for y in vy_sel:
                if y in all_data[fn].columns:
                    trace_name = f"{fn}: {y}"
                    idx = (list(all_data.keys()).index(fn) + vy_sel.index(y)) % 10
                    default_color = px.colors.qualitative.Plotly[idx]
                    custom_colors[trace_name] = st.sidebar.color_picker(f"Color for {trace_name}", default_color, key=f"cp_{trace_name}")

# --- Main App Logic ---
if not all_data:
    st.markdown("<h1 style='text-align:center; color:#1f4e79; margin-top:150px;'>Atlas Analytics Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem; color:#666;'>Secure Multi-tenant SaaS for Biological Data Fitting & Analysis.</p>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ About SaaS Atlas Platform", expanded=True):
        st.markdown("""
        ### Platform Overview
        The **SaaS Atlas Platform** is a professional-grade analytical tool designed for biological and physical data processing. It provides a robust environment for multi-tenant data management, automated curve fitting, and high-dimensional visualization.

        ### Mathematical Fit Functions & Relevance
        1. **Sigmoid Fit:** 
           - *Equation:* $f(t) = a + \\frac{b}{1 + e^{-(t-t_0)/k}}$
           - *Relevance:* Essential for modeling growth phases, cell population dynamics, and saturation processes where a transition between two states occurs.
        2. **Exponential Rise:** 
           - *Equation:* $f(x) = a(1 - e^{-x/b})$
           - *Relevance:* Used for kinetics that approach a steady-state value, such as protein binding or resource-limited growth.
        3. **Linear Regression:** 
           - *Equation:* $y = mx + c$
           - *Relevance:* Determines constant rates of change and baseline drift in experimental sensors.
        4. **Machine Learning (SVR & Random Forest):**
           - *Relevance:* Provides high-fidelity fitting for complex, non-linear biological trends where the underlying mathematical model is not yet defined.
        """)
    st.stop()

# Header with About section on the right
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown(f"<h1 class='main-header'>💠 Atlas Analytics Dashboard</h1>", unsafe_allow_html=True)
with header_col2:
    with st.popover("ℹ️ About Platform"):
        st.markdown("""
        ### SaaS Atlas v3.0
        **Usage:** Multi-tenant data fitting and spatial analysis.
        
        **Functions:**
        - **Sigmoid:** Growth/Saturation.
        - **Exponential:** Rise-to-max.
        - **Linear:** Rate analysis.
        - **ML (SVR/RF):** Complex non-parametric trends.
        
        **Export:** All plots can be exported at up to 1200 DPI to the project `output` folder.
        """)

tabs = st.tabs(["📉 2D Trends", "📊 Statistical Analysis", "🧊 3D Insights", "🔬 Cluster & ML", "📂 Data Manager"])

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import t, ttest_ind, f_oneway
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import os
import io
import warnings
from datetime import datetime
from itertools import combinations

# --- Tab 1: 2D Trends (The Core Plotting) ---
with tabs[0]:
    st.markdown("<h3 class='main-header'>Time-Series & Trend Analysis</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 4])
    
    with c1:
        st.markdown("**Local Settings**")
        p_mode = st.radio("Display Mode", ["Combined (Unified)", "Grid (Split Panels)"])
        m_fit = st.selectbox("Fitting Algorithm", ["None", "Sigmoid", "Exponential", "Linear", "Linear (Advanced)", "ML: Random Forest", "ML: SVR"])
        show_lines = st.checkbox("Connect Data Points (Lines)", value=True)
        
        st.markdown("---")
        show_err_y = st.checkbox("Enable Vertical Error Bars", value=False)
        err_y_col = st.selectbox("Y-Error Source (SD)", ["None"] + all_cols) if show_err_y else None
        
        show_err_x = st.checkbox("Enable Horizontal Error Bars", value=False)
        err_x_col = st.selectbox("X-Error Source (SD)", ["None"] + all_cols) if show_err_x else None
        
        show_ci = st.checkbox("Show CI Band (Linear Advanced)", value=True)
        st.markdown("---")
        save_res = st.selectbox("Resolution (DPI)", [100, 300, 600, 1200], index=1)
        save_format = st.selectbox("Save Format", ["png", "svg", "pdf"])
        export_btn_placeholder = st.empty()
        # save_plot = st.button("💾 Save Plot to Output Folder")

    with c2:
        if not files_to_plot or not vy_sel:
            st.info("Select datasets and Y-axes in the sidebar to begin plotting.")
        else:
            fit_results = []
            valid_x = []
            for fn in files_to_plot:
                if vx in all_data[fn].columns:
                    valid_x.extend(all_data[fn][vx].dropna().tolist())
            global_x_range = (min(valid_x), max(valid_x)) if valid_x else None

            if p_mode == "Combined (Unified)":
                fig_pl = go.Figure()
                fig_mp, ax_mp = plt.subplots(figsize=(12, 7))

                for fn in files_to_plot:
                    for y in vy_sel:
                        if y in all_data[fn].columns and vx in all_data[fn].columns:
                            cols_to_use = [vx, y]
                            if show_err_y and err_y_col in all_data[fn].columns and err_y_col not in cols_to_use: cols_to_use.append(err_y_col)
                            if show_err_x and err_x_col in all_data[fn].columns and err_x_col not in cols_to_use: cols_to_use.append(err_x_col)
                            
                            df = all_data[fn][cols_to_use].copy()
                            if ignore_nan: df = df.dropna()
                            
                            try:
                                if np.issubdtype(df[vx].dtype, np.number):
                                    df = df.sort_values(by=vx)
                            except: pass

                            trace_name = f"{fn}: {y}"
                            t_color = custom_colors.get(trace_name) if use_custom_colors else None

                            # Plotly Trace
                            err_y_pl = dict(type='data', array=df[err_y_col].values, visible=True) if (show_err_y and err_y_col in df.columns) else None
                            err_x_pl = dict(type='data', array=df[err_x_col].values, visible=True) if (show_err_x and err_x_col in df.columns) else None
                            mode_pl = 'lines+markers' if show_lines else 'markers'
                            fig_pl.add_trace(go.Scatter(
                                x=df[vx].values, y=df[y].values, name=trace_name,
                                mode=mode_pl, error_y=err_y_pl, error_x=err_x_pl,
                                marker=dict(size=8, opacity=0.7, color=t_color),
                                line=dict(color=t_color)
                            ))

                            # Matplotlib Trace
                            x_err_mp = df[err_x_col].values if (show_err_x and err_x_col in df.columns) else None
                            y_err_mp = df[err_y_col].values if (show_err_y and err_y_col in df.columns) else None
                            
                            if x_err_mp is not None or y_err_mp is not None:
                                ax_mp.errorbar(df[vx].values, df[y].values, xerr=x_err_mp, yerr=y_err_mp, fmt='o', label=trace_name, alpha=0.5, capsize=3, color=t_color)
                                if show_lines: ax_mp.plot(df[vx].values, df[y].values, alpha=0.7, color=t_color)
                            else:
                                if show_lines:
                                    ax_mp.plot(df[vx].values, df[y].values, marker='o', label=trace_name, alpha=0.7, color=t_color)
                                else:
                                    ax_mp.scatter(df[vx].values, df[y].values, label=trace_name, s=100, alpha=0.7, color=t_color)

                            # Fit Overlay
                            if m_fit != "None":
                                try:
                                    x_p, y_f, p, ci = perform_fit(df[vx], df[y], m_fit, x_range=global_x_range)
                                    if y_f is not None:
                                        fit_label = f"Fit ({m_fit}): {fn}"
                                        fig_pl.add_trace(go.Scatter(x=x_p, y=y_f, name=fit_label, line=dict(dash='solid', width=3, color=t_color)))
                                        ax_mp.plot(x_p, y_f, '--', linewidth=2.5, label=fit_label, color=t_color)
                                        if ci and show_ci:
                                            fig_pl.add_trace(go.Scatter(
                                                x=np.concatenate([x_p, x_p[::-1]]),
                                                y=np.concatenate([ci[0], ci[1][::-1]]),
                                                fill='toself', fillcolor='rgba(0,100,80,0.2)' if not t_color else t_color.replace('rgb', 'rgba').replace(')', ',0.2)'),
                                                line=dict(color='rgba(255,255,255,0)'),
                                                hoverinfo="skip", showlegend=False, name=f"CI: {fn}"
                                            ))
                                            ax_mp.fill_between(x_p, ci[1], ci[0], alpha=0.2, color=t_color if t_color else 'gray')
                                        p.update({"Dataset": fn, "Feature": y, "Model": m_fit})
                                        fit_results.append(p)
                                except: pass
                
                fig_pl.update_layout(
                    xaxis_title=vx, yaxis_title=vy_sel[0] if len(vy_sel)==1 else "Value",
                    showlegend=show_legend,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                    template="plotly_white", hovermode="closest", margin=dict(r=150)
                )
                st.plotly_chart(fig_pl, use_container_width=True)
                
                ax_mp.set_xlabel(vx); ax_mp.set_ylabel(vy_sel[0] if len(vy_sel)==1 else "Value")
                if show_legend: ax_mp.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()

                # Prepare export buffer
                buf = io.BytesIO()
                fig_mp.savefig(buf, format=save_format, dpi=save_res, bbox_inches='tight')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mime="image/png" if save_format == "png" else "image/svg+xml" if save_format == "svg" else "application/pdf"
                export_btn_placeholder.download_button(
                    label=f"💾 Save Plot as {save_format.upper()}",
                    data=buf.getvalue(),
                    file_name=f"2D_Trend_{timestamp}.{save_format}",
                    mime=mime
                )
            else:
                # Grid Mode
                num_plots = len(files_to_plot)
                cols = 2
                rows = (num_plots + 1) // 2
                fig_grid, axes = plt.subplots(rows, cols, figsize=(14, 5*rows), squeeze=False)
                for idx, fn in enumerate(files_to_plot):
                    r, c = idx // cols, idx % cols
                    curr_ax = axes[r, c]
                    local_x_range = (all_data[fn][vx].min(), all_data[fn][vx].max()) if (vx in all_data[fn].columns and np.issubdtype(all_data[fn][vx].dtype, np.number)) else None
                    for y in vy_sel:
                        if y in all_data[fn].columns and vx in all_data[fn].columns:
                            cols_to_use = [vx, y]
                            if show_err_y and err_y_col in all_data[fn].columns:
                                if err_y_col not in cols_to_use: cols_to_use.append(err_y_col)
                            if show_err_x and err_x_col in all_data[fn].columns:
                                if err_x_col not in cols_to_use: cols_to_use.append(err_x_col)
                            
                            df = all_data[fn][cols_to_use].copy()
                            if ignore_nan: df = df.dropna()
                            try:
                                if np.issubdtype(df[vx].dtype, np.number): df = df.sort_values(by=vx)
                            except: pass

                            trace_name = f"{fn}: {y}"
                            t_color = custom_colors.get(trace_name) if use_custom_colors else None

                            x_err_mp = df[err_x_col].values if (show_err_x and err_x_col in df.columns) else None
                            y_err_mp = df[err_y_col].values if (show_err_y and err_y_col in df.columns) else None

                            if x_err_mp is not None or y_err_mp is not None:
                                curr_ax.errorbar(df[vx].values, df[y].values, xerr=x_err_mp, yerr=y_err_mp, fmt='o', label=y, alpha=0.6, capsize=3, color=t_color)
                                if show_lines: curr_ax.plot(df[vx].values, df[y].values, alpha=0.5, color=t_color)
                            else:
                                if show_lines:
                                    curr_ax.plot(df[vx].values, df[y].values, marker='o', label=y, alpha=0.7, color=t_color)
                                else:
                                    curr_ax.scatter(df[vx].values, df[y].values, label=y, s=80, alpha=0.7, color=t_color)
                            
                            if m_fit != "None":
                                try:
                                    x_p, y_f, p, ci = perform_fit(df[vx], df[y], m_fit, x_range=local_x_range)
                                    if y_f is not None:
                                        curr_ax.plot(x_p, y_f, '--', linewidth=2, label=f"Fit: {m_fit}", color=t_color)
                                        if ci and show_ci:
                                            curr_ax.fill_between(x_p, ci[1], ci[0], alpha=0.2, color=t_color if t_color else 'gray')
                                        p.update({"Dataset": fn, "Feature": y, "Model": m_fit})
                                        fit_results.append(p)
                                except: pass
                    curr_ax.set_title(fn); curr_ax.set_xlabel(vx); curr_ax.set_ylabel("Value")
                    if show_legend: curr_ax.legend(loc='best', fontsize='small')
                plt.tight_layout()
                st.pyplot(fig_grid)

                # Prepare export buffer
                buf = io.BytesIO()
                fig_grid.savefig(buf, format=save_format, dpi=save_res, bbox_inches='tight')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_btn_placeholder.download_button(
                    label=f"💾 Save Grid as {save_format.upper()}",
                    data=buf.getvalue(),
                    file_name=f"2D_Grid_{timestamp}.{save_format}",
                    mime=f"image/{save_format}" if save_format != "pdf" else "application/pdf"
                )
            
            if fit_results:
                st.markdown("#### 🔬 Fitting Parameters")
                res_df = pd.DataFrame(fit_results)
                st.dataframe(res_df, use_container_width=True)
                st.download_button("📥 Export Parameters (CSV)", res_df.to_csv(index=False), "fit_results.csv")

# --- Tab 2: Statistical Analysis ---
with tabs[1]:
    st.markdown("<h3 class='main-header'>Distribution & Comparative Analysis</h3>", unsafe_allow_html=True)
    sc1, sc2 = st.columns([1, 4])
    with sc1:
        s_type = st.selectbox("Stat View", ["Statistical Tests ( Welch/ANOVA )", "Boxplot Comparison", "Violin Distribution", "Bar Mean Chart", "Correlation Matrix", "Pie Chart (Relative Volume)"])
        target_y = st.selectbox("Value Column (Numeric)", vy_sel) if vy_sel else None
        stat_format = st.selectbox("Save Format", ["png", "svg", "pdf"], key="stat_format")
        stat_export_placeholder = st.empty()
        # save_stat = st.button("💾 Save Stat Plot")
    with sc2:
        if not files_to_plot or not target_y:
            st.info("Select datasets and a numeric value column to perform analysis.")
        elif s_type == "Statistical Tests ( Welch/ANOVA )":
            st.markdown("#### Pairwise Comparisons & Variance Analysis")
            group_col = st.selectbox("Grouping Column (Categories)", all_cols, index=0)
            test_mode = st.radio("Select Test", ["Welch t-test (Unequal Var)", "Student t-test (Equal Var)", "One-Way ANOVA"], horizontal=True)
            
            # Aggregate data from selected files
            input_table = pd.DataFrame()
            for fn in files_to_plot:
                if all(c in all_data[fn].columns for c in [group_col, target_y]):
                    temp = all_data[fn][[group_col, target_y]].copy()
                    if ignore_nan: temp = temp.dropna()
                    input_table = pd.concat([input_table, temp], ignore_index=True)
            
            if not input_table.empty:
                cell_types = input_table[group_col].unique()
                if len(cell_types) < 2:
                    st.warning("At least 2 unique groups are required in the grouping column.")
                else:
                    if "t-test" in test_mode:
                        result_table = []
                        eq_var = True if "Student" in test_mode else False
                        for c1, c2 in combinations(cell_types, 2):
                            data1 = input_table[input_table[group_col] == c1][target_y]
                            data2 = input_table[input_table[group_col] == c2][target_y]
                            if len(data1) > 1 and len(data2) > 1:
                                t_stat, p_val = ttest_ind(data1, data2, equal_var=eq_var, nan_policy='omit')
                                result_table.append({
                                    'Group 1': c1, 'Group 2': c2,
                                    'Mean 1': np.mean(data1), 'Std 1': np.std(data1), 'n1': len(data1),
                                    'Mean 2': np.mean(data2), 'Std 2': np.std(data2), 'n2': len(data2),
                                    'T-Statistic': t_stat, 'P-Value': p_val
                                })
                        
                        res_df = pd.DataFrame(result_table)
                        st.dataframe(res_df, use_container_width=True)
                        
                        # Build Matrix Map
                        map_df = pd.DataFrame(index=cell_types, columns=cell_types, dtype=float)
                        for _, row in res_df.iterrows():
                            map_df.loc[row['Group 2'], row['Group 1']] = row['P-Value']
                            map_df.loc[row['Group 1'], row['Group 2']] = row['P-Value']
                        
                        st.markdown("#### P-Value Matrix Map")
                        st.dataframe(map_df, use_container_width=True)
                        
                        # Export Options
                        st.markdown("---")
                        c_exp1, c_exp2 = st.columns(2)
                        with c_exp1:
                            export_label = st.text_input("Custom Filename Label", "Supplementary_table")
                        with c_exp2:
                            if st.button("📂 Save Stats to Output Folder"):
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                f1 = os.path.join(OUTPUT_DIR, f"Stats_{export_label}_Full_{ts}.csv")
                                f2 = os.path.join(OUTPUT_DIR, f"Stats_{export_label}_Matrix_{ts}.csv")
                                res_df.to_csv(f1, encoding='cp1252', index=False)
                                map_df.to_csv(f2, encoding='cp1252')
                                st.success(f"Saved: {f1}")
                    
                    elif "ANOVA" in test_mode:
                        groups = [input_table[input_table[group_col] == c][target_y] for c in cell_types]
                        f_stat, p_val = f_oneway(*groups)
                        st.metric("ANOVA F-Statistic", f"{f_stat:.4f}")
                        st.metric("ANOVA P-Value", f"{p_val:.4e}")
            else:
                st.warning("No data found for the selected columns.")
        else:
            # Existing plot-based distributions
            comp_df = pd.DataFrame()
            for fn in files_to_plot:
                if target_y in all_data[fn].columns:
                    temp = all_data[fn][[target_y]].copy()
                    if ignore_nan: temp = temp.dropna()
                    temp['Dataset'] = fn
                    comp_df = pd.concat([comp_df, temp], ignore_index=True)
            
            if not comp_df.empty:
                fig_stat, ax_stat = plt.subplots()
                if s_type == "Boxplot Comparison":
                    sns.boxplot(data=comp_df, x='Dataset', y=target_y, ax=ax_stat)
                elif s_type == "Violin Distribution":
                    sns.violinplot(data=comp_df, x='Dataset', y=target_y, ax=ax_stat)
                elif s_type == "Bar Mean Chart":
                    sns.barplot(data=comp_df, x='Dataset', y=target_y, ax=ax_stat)
                elif s_type == "Correlation Matrix":
                    # Only calculate correlation for numeric columns to avoid ValueError
                    numeric_df = all_data[files_to_plot[0]].select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax_stat)
                    else:
                        st.warning("No numeric columns found for correlation analysis.")
                elif s_type == "Pie Chart (Relative Volume)":
                    sums = [all_data[fn][target_y].sum() for fn in files_to_plot if target_y in all_data[fn].columns]
                    labels = [fn for fn in files_to_plot if target_y in all_data[fn].columns]
                    ax_stat.pie(sums, labels=labels, autopct='%1.1f%%', startangle=140)
                
                plt.xticks(rotation=45)
                st.pyplot(fig_stat)

                # Prepare export buffer
                buf = io.BytesIO()
                fig_stat.savefig(buf, format=stat_format, dpi=300, bbox_inches='tight')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mime="image/png" if stat_format == "png" else "image/svg+xml" if stat_format == "svg" else "application/pdf"
                stat_export_placeholder.download_button(
                    label=f"💾 Save Stat Plot as {stat_format.upper()}",
                    data=buf.getvalue(),
                    file_name=f"Stat_{s_type}_{timestamp}.{stat_format}",
                    mime=mime
                )
# --- Tab 3: 3D Insights ---
with tabs[2]:
    st.markdown("<h3 class='main-header'>Spatial Data Analysis (3D)</h3>", unsafe_allow_html=True)
    c31, c32 = st.columns([1, 4])
    with c31:
        z_ax = st.selectbox("Z-Axis", all_cols, index=min(2, len(all_cols)-1))
        plot_lib = st.selectbox("Plotting Library", ["Plotly (Interactive)", "Matplotlib (Static)"])
        plot_type_3d = st.selectbox("3D Plot Type", ["Scatter", "Surface", "Mesh", "Bar"])
        three_d_format = st.selectbox("Save Format", ["png", "svg", "pdf"], key="3d_format")
        three_d_export_placeholder = st.empty()
        # save_3d = st.button("💾 Save 3D Plot")
    
    with c32:
        if plot_lib == "Plotly (Interactive)":
            fig_3d_pl = go.Figure()
            for fn in files_to_plot:
                df_raw = all_data[fn]
                if all(c in df_raw.columns for c in [vx, vy_sel[0], z_ax]):
                    df = df_raw[[vx, vy_sel[0], z_ax]].copy()
                    if ignore_nan: df = df.dropna()
                    if plot_type_3d == "Scatter":
                        fig_3d_pl.add_trace(go.Scatter3d(x=df[vx].values, y=df[vy_sel[0]].values, z=df[z_ax].values, mode='markers', name=fn, marker=dict(size=5)))
                    elif plot_type_3d == "Mesh":
                        fig_3d_pl.add_trace(go.Mesh3d(x=df[vx].values, y=df[vy_sel[0]].values, z=df[z_ax].values, opacity=0.5, name=fn))
                    elif plot_type_3d == "Bar":
                        # Simplistic 3D Bar representation for Plotly (Scatter3d with lines)
                        fig_3d_pl.add_trace(go.Scatter3d(x=df[vx].values, y=df[vy_sel[0]].values, z=df[z_ax].values, mode='markers+lines', name=fn))
            
            fig_3d_pl.update_layout(
                scene=dict(xaxis_title=vx, yaxis_title=vy_sel[0], zaxis_title=z_ax),
                showlegend=show_legend,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(l=0, r=100, b=0, t=0)
            )
            st.plotly_chart(fig_3d_pl, use_container_width=True)
        else:
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            for fn in files_to_plot:
                df_raw = all_data[fn]
                if all(c in df_raw.columns for c in [vx, vy_sel[0], z_ax]):
                    df = df_raw[[vx, vy_sel[0], z_ax]].copy()
                    if ignore_nan: df = df.dropna()
                    if plot_type_3d == "Scatter":
                        ax_3d.scatter(df[vx].values, df[vy_sel[0]].values, df[z_ax].values, label=fn)
                    elif plot_type_3d == "Bar":
                        ax_3d.bar3d(df[vx].values, df[vy_sel[0]].values, np.zeros_like(df[z_ax].values), 1, 1, df[z_ax].values, shade=True, label=fn)
            ax_3d.set_xlabel(vx); ax_3d.set_ylabel(vy_sel[0]); ax_3d.set_zlabel(z_ax)
            if show_legend: ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_3d)

            # Prepare export buffer
            buf = io.BytesIO()
            fig_3d.savefig(buf, format=three_d_format, dpi=300, bbox_inches='tight')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mime="image/png" if three_d_format == "png" else "image/svg+xml" if three_d_format == "svg" else "application/pdf"
            three_d_export_placeholder.download_button(
                label=f"💾 Save 3D Plot as {three_d_format.upper()}",
                data=buf.getvalue(),
                file_name=f"3D_Plot_{timestamp}.{three_d_format}",
                mime=mime
            )
# --- Tab 4: Cluster & ML ---
with tabs[3]:
    st.markdown("<h3 class='main-header'>Machine Learning & Cluster Analysis</h3>", unsafe_allow_html=True)
    if len(vy_sel) >= 2:
        k_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
        combined_ml = pd.concat([all_data[fn][vy_sel[:2]] for fn in files_to_plot if all(y in all_data[fn].columns for y in vy_sel[:2])], ignore_index=True)
        combined_ml = combined_ml.dropna()
        if not combined_ml.empty:
            kmeans = KMeans(n_clusters=k_clusters, n_init=10).fit(combined_ml)
            combined_ml['Cluster'] = kmeans.labels_
            fig_ml, ax_ml = plt.subplots()
            sns.scatterplot(data=combined_ml, x=vy_sel[0], y=vy_sel[1], hue='Cluster', palette='viridis', ax=ax_ml)
            if not show_legend:
                if ax_ml.get_legend(): ax_ml.get_legend().remove()
            st.pyplot(fig_ml)
        else:
            st.warning("No valid data found for the selected ML features after removing NaNs.")
    else:
        st.warning("Select at least 2 Y-axes features for cluster analysis.")


# --- Tab 5: Data Manager ---
with tabs[4]:
    st.markdown("<h3 class='main-header'>Dataset Inventory</h3>", unsafe_allow_html=True)
    sel_f = st.selectbox("Inspect Dataset", list(all_data.keys()))
    # Data Manager now shows the full table regardless of ignore_nan
    st.dataframe(all_data[sel_f], use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>Atlas SaaS Analytics Platform v3.0 | 2026 Enterprise Edition</p>", unsafe_allow_html=True)
