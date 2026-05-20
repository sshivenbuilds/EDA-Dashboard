import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #111111; }
#MainMenu, footer, header { visibility: hidden !important; }
.stApp { background: #ffffff; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; max-width: 1200px; }
[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; color: #999 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #111 !important; border-bottom: 2px solid #111 !important; }
[data-testid="stFileUploader"] { border: 2px dashed #ddd; border-radius: 12px; padding: 0.5rem; background: #fafafa; }
.metric-card { background: #f8f8f8; border: 1px solid #eee; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.5rem; }
.metric-label { font-size: 0.68rem; color: #999; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'IBM Plex Mono', monospace; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #111; font-family: 'IBM Plex Mono', monospace; margin-top: 0.2rem; }
.sec-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 2px; color: #aaa; text-transform: uppercase; margin-bottom: 0.6rem; margin-top: 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid #f0f0f0; }
.info-box { background: #f8f8f8; border-left: 3px solid #e67e00; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.82rem; color: #555; line-height: 1.6; margin: 0.8rem 0; }
.warn-box { background: #fff8f0; border-left: 3px solid #e67e00; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; font-size: 0.8rem; color: #aa5500; margin: 0.6rem 0; }
.success-box { background: #f0fff4; border-left: 3px solid #2e7d32; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.82rem; color: #2e7d32; line-height: 1.6; margin: 0.8rem 0; }
.outlier-box { background: #fff3f3; border-left: 3px solid #c0392b; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.82rem; color: #c0392b; line-height: 1.6; margin: 0.8rem 0; }
.stSelectbox > div > div { background: #f8f8f8 !important; border: 1px solid #ddd !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


def white_fig(w=7, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#eeeeee')
    ax.tick_params(colors='#666', labelsize=8)
    ax.grid(axis='y', color='#eeeeee', linewidth=0.6, linestyle='--')
    return fig, ax


ORANGE = '#e67e00'
RED    = '#c0392b'
GREEN  = '#2e7d32'

st.markdown("<h2 style='font-family:IBM Plex Mono,monospace;color:#111;margin:0 0 0.2rem 0;'>EDA Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#aaa;font-size:0.84rem;margin:0 0 1rem 0;'>Upload any CSV to instantly explore, analyse and visualise your dataset</p>", unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid #eee;margin-bottom:1rem;'>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0;color:#ccc;">
        <div style="font-size:3rem;margin-bottom:0.8rem;">📂</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;letter-spacing:2px;color:#ccc;">UPLOAD A CSV FILE TO BEGIN</div>
        <div style="margin-top:1rem;font-size:0.76rem;color:#ddd;line-height:2;">
            Works with any dataset · Sales data · Survey results · Stock prices · Sports stats
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    df = pd.read_csv(uploaded)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with q2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with q3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Missing</div><div class="metric-value">{df.isnull().sum().sum():,}</div></div>', unsafe_allow_html=True)
    with q4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Numeric Cols</div><div class="metric-value">{len(num_cols)}</div></div>', unsafe_allow_html=True)
    with q5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Category Cols</div><div class="metric-value">{len(cat_cols)}</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:1rem 0;'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋  Overview", "📊  Statistics", "📈  Distribution",
        "🔗  Correlation", "🥧  Pie Chart", "📉  Line Chart",
        "🔎  Outliers", "🧹  Data Cleaning"
    ])

    # ── TAB 1 — OVERVIEW ──────────────────────────────────────
    with tab1:
        st.markdown("<div class='sec-header'>Dataset Preview (first 10 rows)</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown("<div class='sec-header'>Column Info</div>", unsafe_allow_html=True)
            info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Missing": df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(1)
            })
            st.dataframe(info, use_container_width=True, hide_index=True)
        with right:
            st.markdown("<div class='sec-header'>Missing Values per Column</div>", unsafe_allow_html=True)
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if missing.empty:
                st.markdown('<div class="info-box">✅ No missing values found in this dataset!</div>', unsafe_allow_html=True)
            else:
                try:
                    fig, ax = white_fig(6, max(3, len(missing) * 0.45))
                    ax.barh(missing.index, missing.values, color=ORANGE, alpha=0.85)
                    ax.set_xlabel("Missing Count", fontsize=8, color='#666')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ {e}</div>', unsafe_allow_html=True)

    # ── TAB 2 — STATISTICS ────────────────────────────────────
    with tab2:
        if not num_cols:
            st.warning("No numeric columns found in this dataset.")
        else:
            st.markdown("<div class='sec-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
            stats = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
            stats['Skewness'] = df[num_cols].skew().round(3)
            stats['Kurtosis'] = df[num_cols].kurt().round(3)
            st.dataframe(stats.round(3), use_container_width=True)
            csv_stats = stats.to_csv().encode('utf-8')
            st.download_button(label="⬇️ Download Stats as CSV", data=csv_stats,
                               file_name="summary_statistics.csv", mime="text/csv")
            st.markdown("<div class='sec-header' style='margin-top:1.2rem;'>Per Column Summary</div>", unsafe_allow_html=True)
            cols_per_row = 4
            for i in range(0, len(num_cols), cols_per_row):
                row = st.columns(cols_per_row)
                for j, col in enumerate(num_cols[i:i+cols_per_row]):
                    with row[j]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{col}</div>
                            <div style="margin-top:0.5rem;font-size:0.78rem;color:#555;line-height:1.9;font-family:'IBM Plex Mono',monospace;">
                                Mean &nbsp;&nbsp;&nbsp;: <b>{df[col].mean():,.2f}</b><br>
                                Median &nbsp;: <b>{df[col].median():,.2f}</b><br>
                                Std Dev &nbsp;: <b>{df[col].std():,.2f}</b><br>
                                Min &nbsp;&nbsp;&nbsp;&nbsp;: <b>{df[col].min():,.2f}</b><br>
                                Max &nbsp;&nbsp;&nbsp;&nbsp;: <b>{df[col].max():,.2f}</b><br>
                                Skewness: <b>{df[col].skew():,.3f}</b><br>
                                Kurtosis: <b>{df[col].kurt():,.3f}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── TAB 3 — DISTRIBUTION ──────────────────────────────────
    with tab3:
        if not num_cols:
            st.warning("No numeric columns available for distribution plots.")
        else:
            st.markdown("<div class='sec-header'>Select Column</div>", unsafe_allow_html=True)
            col_pick = st.selectbox("Column to plot", options=num_cols, label_visibility="collapsed", key="dist_col")
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"<div class='sec-header'>Histogram — {col_pick}</div>", unsafe_allow_html=True)
                try:
                    fig, ax = white_fig(6, 4)
                    ax.hist(df[col_pick].dropna(), bins=30, color=ORANGE, alpha=0.85, edgecolor='white', linewidth=0.5)
                    ax.set_xlabel(col_pick, fontsize=9, color='#555')
                    ax.set_ylabel("Frequency", fontsize=9, color='#555')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ Could not plot histogram: {e}</div>', unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div class='sec-header'>Box Plot — {col_pick}</div>", unsafe_allow_html=True)
                try:
                    fig, ax = white_fig(6, 4)
                    ax.boxplot(df[col_pick].dropna(), patch_artist=True,
                               boxprops=dict(facecolor='#fff3e0', color=ORANGE),
                               medianprops=dict(color=ORANGE, linewidth=2),
                               whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
                               flierprops=dict(marker='o', color=ORANGE, alpha=0.4, markersize=4))
                    ax.set_ylabel(col_pick, fontsize=9, color='#555')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ Could not plot box plot: {e}</div>', unsafe_allow_html=True)

            st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:0.8rem 0;'>", unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value" style="font-size:1.1rem;">{df[col_pick].mean():,.2f}</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Median</div><div class="metric-value" style="font-size:1.1rem;">{df[col_pick].median():,.2f}</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value" style="font-size:1.1rem;">{df[col_pick].std():,.2f}</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Min</div><div class="metric-value" style="font-size:1.1rem;">{df[col_pick].min():,.2f}</div></div>', unsafe_allow_html=True)
            with s5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Max</div><div class="metric-value" style="font-size:1.1rem;">{df[col_pick].max():,.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>Scatter Plot — Choose X and Y Axis</div>", unsafe_allow_html=True)
            if len(num_cols) < 2:
                st.markdown('<div class="warn-box">⚠️ Need at least 2 numeric columns for a scatter plot.</div>', unsafe_allow_html=True)
            else:
                sc1, sc2 = st.columns(2)
                with sc1:
                    x_col = st.selectbox("X-axis", options=num_cols, key="scatter_x")
                with sc2:
                    y_col = st.selectbox("Y-axis", options=num_cols, index=min(1, len(num_cols)-1), key="scatter_y")
                try:
                    fig, ax = white_fig(10, 4)
                    ax.scatter(df[x_col].dropna(), df[y_col].dropna(), color=ORANGE, alpha=0.5, s=20, edgecolors='none')
                    ax.set_xlabel(x_col, fontsize=9, color='#555')
                    ax.set_ylabel(y_col, fontsize=9, color='#555')
                    ax.set_title(f"{x_col}  vs  {y_col}", fontsize=10, color='#333', pad=8)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ Could not plot scatter: {e}</div>', unsafe_allow_html=True)

    # ── TAB 4 — CORRELATION ───────────────────────────────────
    with tab4:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            st.markdown("<div class='sec-header'>Correlation Method</div>", unsafe_allow_html=True)
            method_display = ["Pearson", "Spearman", "Kendall Tau"]
            method_map = {"Pearson": "pearson", "Spearman": "spearman", "Kendall Tau": "kendall"}
            corr_choice = st.radio("", options=method_display, horizontal=True, key="corr_method", label_visibility="collapsed")
            corr_method = method_map[corr_choice]
            st.markdown("<div class='sec-header'>Select Columns for Heatmap</div>", unsafe_allow_html=True)
            selected_corr = st.multiselect("Choose columns (2 or more)", options=num_cols,
                                           default=num_cols[:min(6, len(num_cols))], key="corr_cols")
            if len(selected_corr) < 2:
                st.markdown('<div class="warn-box">⚠️ Please select at least 2 columns.</div>', unsafe_allow_html=True)
            else:
                try:
                    corr = df[selected_corr].corr(method=corr_method)
                    fig, ax = plt.subplots(figsize=(max(6, len(selected_corr)), max(5, len(selected_corr) * 0.8)))
                    fig.patch.set_facecolor('#ffffff')
                    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax,
                                cmap=sns.diverging_palette(20, 220, as_cmap=True),
                                linewidths=0.5, linecolor='white', annot_kws={"size": 8},
                                vmin=-1, vmax=1, square=True)
                    ax.tick_params(colors='#444', labelsize=8)
                    ax.set_title(f"{corr_choice} Correlation Matrix", fontsize=11, color='#333', pad=12)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown('<div class="info-box">Values close to <b>+1</b> = strong positive correlation · Close to <b>-1</b> = strong negative · Close to <b>0</b> = no relationship.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ Could not generate heatmap: {e}</div>', unsafe_allow_html=True)

    # ── TAB 5 — PIE CHART ─────────────────────────────────────
    with tab5:
        st.markdown("<div class='sec-header'>Select Column for Pie Chart</div>", unsafe_allow_html=True)
        pie_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 20]
        if not pie_options:
            st.warning("No suitable columns found. Pie chart works best with categorical columns or numeric columns with 20 or fewer unique values.")
        else:
            pie_col = st.selectbox("Column", options=pie_options, key="pie_col", label_visibility="collapsed")
            val_counts = df[pie_col].value_counts()
            total_unique = len(val_counts)
            if total_unique > 12:
                st.markdown(f'<div class="warn-box">⚠️ {total_unique} unique values found — showing top 10 + Others.</div>', unsafe_allow_html=True)
                top = val_counts.head(10)
                others = pd.Series([val_counts[10:].sum()], index=["Others"])
                val_counts = pd.concat([top, others])
            try:
                colors = ['#e67e00','#333333','#888888','#f0a040','#555555','#ffcc80','#aaaaaa','#cc6600','#dddddd','#222222','#ffaa33']
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#ffffff')
                wedges, texts, autotexts = ax.pie(
                    val_counts.values, labels=val_counts.index, autopct='%1.1f%%',
                    colors=colors[:len(val_counts)], startangle=90,
                    wedgeprops=dict(edgecolor='white', linewidth=1.5))
                for t in texts: t.set_fontsize(8); t.set_color('#333')
                for at in autotexts: at.set_fontsize(7); at.set_color('white')
                ax.set_title(f"Distribution of '{pie_col}'", fontsize=11, color='#333', pad=12)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown("<div class='sec-header' style='margin-top:1rem;'>Value Counts</div>", unsafe_allow_html=True)
                vc_df = val_counts.reset_index()
                vc_df.columns = [pie_col, 'Count']
                vc_df['Percentage'] = (vc_df['Count'] / vc_df['Count'].sum() * 100).round(1).astype(str) + '%'
                st.dataframe(vc_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.markdown(f'<div class="warn-box">⚠️ Could not generate pie chart: {e}</div>', unsafe_allow_html=True)

    # ── TAB 6 — LINE CHART ────────────────────────────────────
    with tab6:
        if not num_cols:
            st.warning("No numeric columns available for line chart.")
        else:
            st.markdown("<div class='sec-header'>Select Column for Line Chart</div>", unsafe_allow_html=True)
            lc1, lc2 = st.columns(2)
            with lc1:
                y_line = st.selectbox("Y-axis (value to plot)", options=num_cols, key="line_y")
            with lc2:
                x_options = ["Row Index"] + df.columns.tolist()
                x_line = st.selectbox("X-axis (use index or date column)", options=x_options, key="line_x")
            try:
                fig, ax = white_fig(10, 4.5)
                if x_line == "Row Index":
                    ax.plot(df[y_line].values, color=ORANGE, linewidth=1.4)
                    ax.set_xlabel("Row Index", fontsize=9, color='#555')
                else:
                    ax.plot(df[x_line].astype(str), df[y_line].values, color=ORANGE, linewidth=1.4)
                    ax.set_xlabel(x_line, fontsize=9, color='#555')
                    n = max(1, len(df) // 10)
                    ax.set_xticks(range(0, len(df), n))
                    ax.set_xticklabels([str(df[x_line].iloc[i]) for i in range(0, len(df), n)], rotation=30, fontsize=7)
                ax.set_ylabel(y_line, fontsize=9, color='#555')
                ax.set_title(f"{y_line} over {x_line}", fontsize=10, color='#333', pad=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:0.8rem 0;'>", unsafe_allow_html=True)
                l1, l2, l3, l4, l5, l6, l7 = st.columns(7)
                with l1: st.markdown(f'<div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value" style="font-size:1rem;">{df[y_line].mean():,.2f}</div></div>', unsafe_allow_html=True)
                with l2: st.markdown(f'<div class="metric-card"><div class="metric-label">Median</div><div class="metric-value" style="font-size:1rem;">{df[y_line].median():,.2f}</div></div>', unsafe_allow_html=True)
                with l3: st.markdown(f'<div class="metric-card"><div class="metric-label">Min</div><div class="metric-value" style="font-size:1rem;">{df[y_line].min():,.2f}</div></div>', unsafe_allow_html=True)
                with l4: st.markdown(f'<div class="metric-card"><div class="metric-label">Max</div><div class="metric-value" style="font-size:1rem;">{df[y_line].max():,.2f}</div></div>', unsafe_allow_html=True)
                with l5: st.markdown(f'<div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value" style="font-size:1rem;">{df[y_line].std():,.2f}</div></div>', unsafe_allow_html=True)
                with l6: st.markdown(f'<div class="metric-card"><div class="metric-label">Skewness</div><div class="metric-value" style="font-size:1rem;">{df[y_line].skew():,.3f}</div></div>', unsafe_allow_html=True)
                with l7: st.markdown(f'<div class="metric-card"><div class="metric-label">Kurtosis</div><div class="metric-value" style="font-size:1rem;">{df[y_line].kurt():,.3f}</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="warn-box">⚠️ Could not plot line chart: {e}</div>', unsafe_allow_html=True)

    # ── TAB 7 — OUTLIERS ──────────────────────────────────────
    with tab7:
        if not num_cols:
            st.warning("No numeric columns available for outlier detection.")
        else:
            st.markdown("<div class='sec-header'>Select Column</div>", unsafe_allow_html=True)
            out_col = st.selectbox("Column to analyse", options=num_cols, key="out_col", label_visibility="collapsed")

            # Auto-recommend method based on skewness
            skew_val = df[out_col].skew()
            if abs(skew_val) > 1:
                recommended = "IQR"
                reason = f"Skewness is {skew_val:.2f} (highly skewed) → IQR method recommended"
            else:
                recommended = "Z-Score"
                reason = f"Skewness is {skew_val:.2f} (near normal) → Z-Score method recommended"

            st.markdown(f'<div class="info-box">🧠 <b>Auto Recommendation:</b> {reason}</div>', unsafe_allow_html=True)

            st.markdown("<div class='sec-header'>Choose Detection Method</div>", unsafe_allow_html=True)
            method_out = st.radio("", options=["IQR Method", "Z-Score Method"],
                                  horizontal=True, key="out_method", label_visibility="collapsed",
                                  index=0 if recommended == "IQR" else 1)

            col_data = df[out_col].dropna()

            if method_out == "IQR Method":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask = (df[out_col] < lower) | (df[out_col] > upper)
                method_info = f"Q1={Q1:.2f} · Q3={Q3:.2f} · IQR={IQR:.2f} · Lower fence={lower:.2f} · Upper fence={upper:.2f}"
            else:
                z_scores = np.abs(scipy_stats.zscore(col_data))
                z_full = pd.Series(index=df.index, dtype=float)
                z_full[col_data.index] = np.abs(scipy_stats.zscore(col_data))
                outlier_mask = z_full > 3
                lower, upper = col_data.mean() - 3*col_data.std(), col_data.mean() + 3*col_data.std()
                method_info = f"Mean={col_data.mean():.2f} · Std={col_data.std():.2f} · Threshold = |Z| > 3"

            outlier_df = df[outlier_mask]
            clean_df_preview = df[~outlier_mask]
            n_outliers = len(outlier_df)
            pct = (n_outliers / len(df)) * 100

            # Stats row
            st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:0.8rem 0;'>", unsafe_allow_html=True)
            oc1, oc2, oc3, oc4 = st.columns(4)
            with oc1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Rows</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
            with oc2: st.markdown(f'<div class="metric-card"><div class="metric-label">Outliers Found</div><div class="metric-value" style="color:#c0392b;">{n_outliers:,}</div></div>', unsafe_allow_html=True)
            with oc3: st.markdown(f'<div class="metric-card"><div class="metric-label">Outlier %</div><div class="metric-value" style="color:#c0392b;">{pct:.1f}%</div></div>', unsafe_allow_html=True)
            with oc4: st.markdown(f'<div class="metric-card"><div class="metric-label">Clean Rows</div><div class="metric-value" style="color:#2e7d32;">{len(clean_df_preview):,}</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="info-box">📐 {method_out}: {method_info}</div>', unsafe_allow_html=True)

            # Visualise outliers on box plot
            st.markdown("<div class='sec-header'>Outlier Visualisation</div>", unsafe_allow_html=True)
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#ffffff')

                # Box plot with outliers highlighted
                axes[0].set_facecolor('#f8f8f8')
                bp = axes[0].boxplot(col_data, patch_artist=True,
                                     boxprops=dict(facecolor='#fff3e0', color=ORANGE),
                                     medianprops=dict(color=ORANGE, linewidth=2),
                                     whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
                                     flierprops=dict(marker='o', color=RED, markersize=5, alpha=0.6))
                axes[0].set_title(f"Box Plot — {out_col}", fontsize=10, color='#333')
                axes[0].axhline(y=lower, color=RED, linestyle='--', linewidth=1, alpha=0.7, label=f'Lower: {lower:.2f}')
                axes[0].axhline(y=upper, color=RED, linestyle='--', linewidth=1, alpha=0.7, label=f'Upper: {upper:.2f}')
                axes[0].legend(fontsize=7, facecolor='white')
                for spine in axes[0].spines.values(): spine.set_edgecolor('#eee')
                axes[0].tick_params(colors='#666', labelsize=8)

                # Histogram with outlier zones shaded
                axes[1].set_facecolor('#f8f8f8')
                axes[1].hist(col_data, bins=30, color=ORANGE, alpha=0.7, edgecolor='white')
                axes[1].axvline(x=lower, color=RED, linestyle='--', linewidth=1.5, label=f'Lower: {lower:.2f}')
                axes[1].axvline(x=upper, color=RED, linestyle='--', linewidth=1.5, label=f'Upper: {upper:.2f}')
                axes[1].axvspan(col_data.min(), lower, alpha=0.1, color=RED)
                axes[1].axvspan(upper, col_data.max(), alpha=0.1, color=RED)
                axes[1].set_title(f"Histogram with Outlier Zones — {out_col}", fontsize=10, color='#333')
                axes[1].legend(fontsize=7, facecolor='white')
                axes[1].set_xlabel(out_col, fontsize=8, color='#666')
                for spine in axes[1].spines.values(): spine.set_edgecolor('#eee')
                axes[1].tick_params(colors='#666', labelsize=8)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.markdown(f'<div class="warn-box">⚠️ {e}</div>', unsafe_allow_html=True)

            # Show outlier rows
            if n_outliers > 0:
                st.markdown("<div class='sec-header'>Outlier Rows</div>", unsafe_allow_html=True)
                st.dataframe(outlier_df, use_container_width=True)
                csv_out = outlier_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Outlier Rows", csv_out, "outliers.csv", "text/csv")
            else:
                st.markdown(f'<div class="success-box">✅ No outliers found in <b>{out_col}</b> using {method_out}!</div>', unsafe_allow_html=True)

    # ── TAB 8 — DATA CLEANING ─────────────────────────────────
    with tab8:
        st.markdown("<div class='sec-header'>Data Cleaning Options</div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box">🧹 Apply one or more cleaning steps below. Each step works on the <b>original uploaded data</b>. Download the cleaned file at the bottom.</div>', unsafe_allow_html=True)

        df_clean = df.copy()

        # ── Section 1: Handle Missing Values ──
        st.markdown("<div class='sec-header'>Step 1 — Handle Missing Values</div>", unsafe_allow_html=True)
        missing_total = df_clean.isnull().sum().sum()
        if missing_total == 0:
            st.markdown('<div class="success-box">✅ No missing values in this dataset — nothing to handle!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">⚠️ Found <b>{missing_total}</b> missing values across the dataset.</div>', unsafe_allow_html=True)
            missing_strategy = st.radio(
                "How to handle missing values?",
                options=["Keep as is", "Drop rows with any missing value",
                         "Fill numeric with Mean", "Fill numeric with Median",
                         "Fill numeric with 0"],
                horizontal=False, key="missing_strategy"
            )
            if missing_strategy == "Drop rows with any missing value":
                before = len(df_clean)
                df_clean = df_clean.dropna()
                st.markdown(f'<div class="success-box">✅ Dropped {before - len(df_clean)} rows. {len(df_clean)} rows remaining.</div>', unsafe_allow_html=True)
            elif missing_strategy == "Fill numeric with Mean":
                for col in df_clean.select_dtypes(include=[np.number]).columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                st.markdown('<div class="success-box">✅ Filled all numeric missing values with column mean.</div>', unsafe_allow_html=True)
            elif missing_strategy == "Fill numeric with Median":
                for col in df_clean.select_dtypes(include=[np.number]).columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                st.markdown('<div class="success-box">✅ Filled all numeric missing values with column median.</div>', unsafe_allow_html=True)
            elif missing_strategy == "Fill numeric with 0":
                for col in df_clean.select_dtypes(include=[np.number]).columns:
                    df_clean[col] = df_clean[col].fillna(0)
                st.markdown('<div class="success-box">✅ Filled all numeric missing values with 0.</div>', unsafe_allow_html=True)

        # ── Section 2: Remove Outliers ──
        st.markdown("<div class='sec-header'>Step 2 — Remove Outliers</div>", unsafe_allow_html=True)
        num_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols_clean:
            st.markdown('<div class="warn-box">⚠️ No numeric columns available.</div>', unsafe_allow_html=True)
        else:
            remove_outliers = st.checkbox("Remove outliers from numeric columns", key="remove_outliers")
            if remove_outliers:
                outlier_cols = st.multiselect("Select columns to remove outliers from",
                                              options=num_cols_clean,
                                              default=num_cols_clean[:min(3, len(num_cols_clean))],
                                              key="clean_out_cols")
                out_method_clean = st.radio("Method", options=["IQR Method", "Z-Score Method"],
                                            horizontal=True, key="clean_out_method")

                if outlier_cols:
                    before = len(df_clean)
                    for col in outlier_cols:
                        col_data = df_clean[col].dropna()
                        skew = col_data.skew()

                        if out_method_clean == "IQR Method":
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
                        else:
                            z = np.abs(scipy_stats.zscore(df_clean[col].dropna()))
                            z_full = pd.Series(np.nan, index=df_clean.index)
                            z_full[df_clean[col].dropna().index] = z
                            df_clean = df_clean[z_full <= 3]

                    removed = before - len(df_clean)
                    st.markdown(f'<div class="success-box">✅ Removed <b>{removed}</b> outlier rows using {out_method_clean}. <b>{len(df_clean)}</b> rows remaining.</div>', unsafe_allow_html=True)

        # ── Section 3: Drop Columns ──
        st.markdown("<div class='sec-header'>Step 3 — Drop Columns</div>", unsafe_allow_html=True)
        drop_cols = st.multiselect("Select columns to drop", options=df_clean.columns.tolist(), key="drop_cols")
        if drop_cols:
            df_clean = df_clean.drop(columns=drop_cols)
            st.markdown(f'<div class="success-box">✅ Dropped columns: <b>{", ".join(drop_cols)}</b></div>', unsafe_allow_html=True)

        # ── Section 4: Drop Duplicates ──
        st.markdown("<div class='sec-header'>Step 4 — Remove Duplicate Rows</div>", unsafe_allow_html=True)
        n_dupes = df_clean.duplicated().sum()
        if n_dupes == 0:
            st.markdown('<div class="success-box">✅ No duplicate rows found.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">⚠️ Found <b>{n_dupes}</b> duplicate rows.</div>', unsafe_allow_html=True)
            if st.checkbox("Remove duplicate rows", key="remove_dupes"):
                df_clean = df_clean.drop_duplicates()
                st.markdown(f'<div class="success-box">✅ Removed {n_dupes} duplicate rows. {len(df_clean)} rows remaining.</div>', unsafe_allow_html=True)

        # ── Final Preview + Download ──
        st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:1.2rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-header'>Cleaned Dataset Preview</div>", unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        with fc1: st.markdown(f'<div class="metric-card"><div class="metric-label">Original Rows</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
        with fc2: st.markdown(f'<div class="metric-card"><div class="metric-label">Clean Rows</div><div class="metric-value" style="color:#2e7d32;">{len(df_clean):,}</div></div>', unsafe_allow_html=True)
        with fc3: st.markdown(f'<div class="metric-card"><div class="metric-label">Rows Removed</div><div class="metric-value" style="color:#c0392b;">{len(df) - len(df_clean):,}</div></div>', unsafe_allow_html=True)

        st.dataframe(df_clean.head(10), use_container_width=True)

        csv_clean = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned Dataset as CSV",
            data=csv_clean,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
