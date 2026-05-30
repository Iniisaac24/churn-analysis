# ============================================================
# Churn Analysis Dashboard — Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Churn Analysis",
    page_icon  = "📉",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #ffffff; }
    .block-container { padding: 2rem 2.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f7f7f5;
        border: 1px solid #e8e8e4;
        border-radius: 8px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: .07em !important;
        text-transform: uppercase !important;
        color: #b8b8b4 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #111110 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: #e8600a;
        margin-bottom: 4px;
        margin-top: 32px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111110;
        margin-bottom: 20px;
        letter-spacing: -.01em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f7f7f5;
        border-right: 1px solid #e8e8e4;
    }
    [data-testid="stSidebar"] .block-container { padding: 1.5rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #e8e8e4;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        font-weight: 500;
        color: #737370;
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #111110 !important;
        border-bottom: 2px solid #e8600a !important;
    }

    /* Divider */
    hr { border-color: #e8e8e4; margin: 24px 0; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn_binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['service_count'] = df[[
        'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]].apply(lambda row: sum(1 for v in row if v == 'Yes'), axis=1)
    df['tenure_band'] = pd.cut(
        df['tenure'],
        bins=[0,12,24,48,72],
        labels=['0–12m','13–24m','25–48m','49–72m']
    )
    return df

df = load_data()

# ── SHAP feature importance (from your results) ───────────────
shap_data = pd.DataFrame({
    'Feature': [
        'Contract: Two Year', 'Tenure', 'Internet: Fiber Optic',
        'Total Charges', 'Monthly Charges', 'Contract: One Year',
        'Payment: Electronic Check', 'Paperless Billing',
        'Online Security', 'Multiple Lines'
    ],
    'SHAP Value': [0.5957, 0.5147, 0.4063, 0.3433, 0.3111,
                   0.2878, 0.1999, 0.1355, 0.1265, 0.1223]
})

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.markdown("---")

    contract_filter = st.multiselect(
        "Contract Type",
        options = df['Contract'].unique().tolist(),
        default = df['Contract'].unique().tolist()
    )
    internet_filter = st.multiselect(
        "Internet Service",
        options = df['InternetService'].unique().tolist(),
        default = df['InternetService'].unique().tolist()
    )
    tenure_filter = st.slider(
        "Tenure Range (months)",
        min_value = 0,
        max_value = 72,
        value     = (0, 72)
    )

    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown("""
    <div style='background:#fff8f5;border:1px solid #fde0cc;
                border-radius:8px;padding:14px;font-size:13px;'>
        <div style='color:#e8600a;font-weight:600;
                    font-size:10px;letter-spacing:.08em;
                    text-transform:uppercase;margin-bottom:8px'>
            XGBoost
        </div>
        <div style='display:flex;justify-content:space-between;
                    margin-bottom:6px'>
            <span style='color:#737370'>AUC-ROC</span>
            <span style='font-weight:600;color:#111110'>0.8372</span>
        </div>
        <div style='display:flex;justify-content:space-between;
                    margin-bottom:6px'>
            <span style='color:#737370'>Churn Recall</span>
            <span style='font-weight:600;color:#111110'>76%</span>
        </div>
        <div style='display:flex;justify-content:space-between;
                    margin-bottom:6px'>
            <span style='color:#737370'>Churn Precision</span>
            <span style='font-weight:600;color:#111110'>54%</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#737370'>Training set</span>
            <span style='font-weight:600;color:#111110'>5,634 rows</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("IBM Telco Customer Churn Dataset · 7,043 customers")

# ── Filter data ───────────────────────────────────────────────
dff = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter)) &
    (df['tenure'].between(tenure_filter[0], tenure_filter[1]))
]

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
    <span style='font-size:11px;font-weight:600;letter-spacing:.09em;
                 text-transform:uppercase;color:#e8600a'>
        Telecom · Predictive Analytics
    </span>
</div>
<h1 style='font-size:32px;font-weight:700;color:#111110;
           letter-spacing:-.02em;margin:0 0 6px 0'>
    Customer Churn Analysis
</h1>
<p style='font-size:15px;color:#737370;font-weight:300;
          margin-bottom:28px;'>
    XGBoost model identifying at-risk customers — retention campaign
    projected to cut annual churn by 22%.
</p>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total       = len(dff)
churned     = dff['Churn_binary'].sum()
churn_rate  = dff['Churn_binary'].mean()
avg_charge  = dff[dff['Churn_binary']==1]['MonthlyCharges'].mean()
avg_tenure  = dff[dff['Churn_binary']==1]['tenure'].median()

k1.metric("Total Customers",   f"{total:,}")
k2.metric("Churned",           f"{churned:,}",    f"{churn_rate:.1%} of base")
k3.metric("Avg Charge (Churned)", f"${avg_charge:.2f}", "vs $61.27 retained")
k4.metric("Median Tenure (Churned)", f"{avg_tenure:.0f} mo", "vs 38mo retained")

st.markdown("<hr>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Exploratory Analysis",
    "🤖  Model Results",
    "💰  Business Impact"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Exploratory Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">What drives churn in this dataset?</p>', unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        # Churn by contract
        contract_churn = dff.groupby('Contract')['Churn_binary'].mean().reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate']
        contract_churn['Churn Rate %'] = (contract_churn['Churn Rate'] * 100).round(1)

        fig = px.bar(
            contract_churn, x='Contract', y='Churn Rate',
            text='Churn Rate %',
            color='Churn Rate',
            color_continuous_scale=[[0,'#fde8d8'],[0.5,'#f97316'],[1,'#e8600a']],
            title='Churn Rate by Contract Type'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            yaxis=dict(tickformat='.0%', gridcolor='#e8e8e4'),
            xaxis=dict(gridcolor='white'),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Churn by internet service
        internet_churn = dff.groupby('InternetService')['Churn_binary'].mean().reset_index()
        internet_churn.columns = ['Internet Service', 'Churn Rate']
        internet_churn['Churn Rate %'] = (internet_churn['Churn Rate'] * 100).round(1)

        fig2 = px.bar(
            internet_churn, x='Internet Service', y='Churn Rate',
            text='Churn Rate %',
            color='Churn Rate',
            color_continuous_scale=[[0,'#fde8d8'],[0.5,'#f97316'],[1,'#e8600a']],
            title='Churn Rate by Internet Service'
        )
        fig2.update_traces(texttemplate='%{text}%', textposition='outside')
        fig2.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            yaxis=dict(tickformat='.0%', gridcolor='#e8e8e4'),
            xaxis=dict(gridcolor='white'),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        # Tenure distribution
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=dff[dff['Churn_binary']==0]['tenure'],
            name='Retained', opacity=0.7,
            marker_color='#22c55e', nbinsx=30
        ))
        fig3.add_trace(go.Histogram(
            x=dff[dff['Churn_binary']==1]['tenure'],
            name='Churned', opacity=0.7,
            marker_color='#e8600a', nbinsx=30
        ))
        fig3.update_layout(
            barmode='overlay', title='Tenure Distribution',
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            legend=dict(orientation='h', y=1.1),
            xaxis=dict(title='Tenure (months)', gridcolor='#e8e8e4'),
            yaxis=dict(title='Count', gridcolor='#e8e8e4'),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Monthly charges box
        fig4 = go.Figure()
        fig4.add_trace(go.Box(
            y=dff[dff['Churn_binary']==0]['MonthlyCharges'],
            name='Retained', marker_color='#22c55e',
            boxmean=True
        ))
        fig4.add_trace(go.Box(
            y=dff[dff['Churn_binary']==1]['MonthlyCharges'],
            name='Churned', marker_color='#e8600a',
            boxmean=True
        ))
        fig4.update_layout(
            title='Monthly Charges Distribution',
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            yaxis=dict(title='Monthly Charges ($)', gridcolor='#e8e8e4'),
            showlegend=True,
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3
    c5, c6 = st.columns(2)

    with c5:
        # Churn by service count
        svc_churn = dff.groupby('service_count')['Churn_binary'].mean().reset_index()
        svc_churn.columns = ['Services Subscribed', 'Churn Rate']
        fig5 = px.line(
            svc_churn, x='Services Subscribed', y='Churn Rate',
            markers=True, title='Churn Rate by Number of Services',
            color_discrete_sequence=['#e8600a']
        )
        fig5.update_traces(marker=dict(size=8))
        fig5.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            yaxis=dict(tickformat='.0%', gridcolor='#e8e8e4'),
            xaxis=dict(gridcolor='#e8e8e4'),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        # Payment method
        pay_churn = dff.groupby('PaymentMethod')['Churn_binary'].mean().reset_index()
        pay_churn.columns = ['Payment Method', 'Churn Rate']
        pay_churn = pay_churn.sort_values('Churn Rate', ascending=True)
        fig6 = px.bar(
            pay_churn, x='Churn Rate', y='Payment Method',
            orientation='h', title='Churn Rate by Payment Method',
            color='Churn Rate',
            color_continuous_scale=[[0,'#fde8d8'],[1,'#e8600a']]
        )
        fig6.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            xaxis=dict(tickformat='.0%', gridcolor='#e8e8e4'),
            yaxis=dict(gridcolor='white'),
            margin=dict(t=40, b=20, l=10)
        )
        st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Model Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">XGBoost outperforms on every metric that matters.</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",        "0.837", "vs 0.815 Logistic Reg")
    m2.metric("Churn Recall",   "76%",   "+13pp vs Logistic Reg")
    m3.metric("Churn Precision","54%",   "Acceptable for campaign")
    m4.metric("Overall Accuracy","76%",  "Both models equal here")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # SHAP feature importance bar
        fig_shap = px.bar(
            shap_data.sort_values('SHAP Value'),
            x='SHAP Value', y='Feature',
            orientation='h',
            title='Top 10 Churn Drivers — SHAP Values',
            color='SHAP Value',
            color_continuous_scale=[[0,'#fde8d8'],[1,'#e8600a']]
        )
        fig_shap.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            xaxis=dict(title='Mean |SHAP Value|', gridcolor='#e8e8e4'),
            yaxis=dict(gridcolor='white'),
            margin=dict(t=40, b=20, l=10)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col2:
        # Model comparison bar
        compare_df = pd.DataFrame({
            'Metric' : ['AUC-ROC','Churn Recall','Churn Precision','Churn F1'],
            'Logistic Regression': [0.815, 0.63, 0.55, 0.59],
            'XGBoost':             [0.837, 0.76, 0.54, 0.63]
        })
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name='Logistic Regression',
            x=compare_df['Metric'],
            y=compare_df['Logistic Regression'],
            marker_color='#e8e8e4'
        ))
        fig_cmp.add_trace(go.Bar(
            name='XGBoost',
            x=compare_df['Metric'],
            y=compare_df['XGBoost'],
            marker_color='#e8600a'
        ))
        fig_cmp.update_layout(
            barmode='group',
            title='Model Comparison',
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#111110'),
            font=dict(color='#737370', size=12),
            yaxis=dict(range=[0,1], gridcolor='#e8e8e4'),
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Key insights
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div style='background:#f7f7f5;border:1px solid #e8e8e4;
                    border-radius:8px;padding:20px;height:130px'>
            <div style='font-size:11px;font-weight:600;letter-spacing:.07em;
                        text-transform:uppercase;color:#e8600a;margin-bottom:8px'>
                Top Driver
            </div>
            <div style='font-size:15px;font-weight:600;color:#111110;
                        margin-bottom:6px'>Contract Type</div>
            <div style='font-size:13px;color:#737370;line-height:1.6'>
                Month-to-month customers churn at 42.7% — 15× the rate
                of two-year contract holders.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div style='background:#f7f7f5;border:1px solid #e8e8e4;
                    border-radius:8px;padding:20px;height:130px'>
            <div style='font-size:11px;font-weight:600;letter-spacing:.07em;
                        text-transform:uppercase;color:#e8600a;margin-bottom:8px'>
                Early Warning
            </div>
            <div style='font-size:15px;font-weight:600;color:#111110;
                        margin-bottom:6px'>First 12 Months</div>
            <div style='font-size:13px;color:#737370;line-height:1.6'>
                Median churn tenure is 10 months. Customers who leave
                do so early — onboarding is critical.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div style='background:#f7f7f5;border:1px solid #e8e8e4;
                    border-radius:8px;padding:20px;height:130px'>
            <div style='font-size:11px;font-weight:600;letter-spacing:.07em;
                        text-transform:uppercase;color:#e8600a;margin-bottom:8px'>
                Price Signal
            </div>
            <div style='font-size:15px;font-weight:600;color:#111110;
                        margin-bottom:6px'>Fiber Optic Risk</div>
            <div style='font-size:13px;color:#737370;line-height:1.6'>
                Fiber optic customers pay more and churn more.
                Third strongest SHAP driver after contract and tenure.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS IMPACT
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Business Impact</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">What is the retention campaign worth?</p>', unsafe_allow_html=True)

    st.markdown("#### Adjust the campaign parameters")
    b1, b2, b3 = st.columns(3)

    with b1:
        offer_cost = st.slider(
            "Retention offer cost per customer ($)",
            min_value = 5,
            max_value = 50,
            value     = 10,
            step      = 5
        )
    with b2:
        retention_rate = st.slider(
            "Assumed retention rate (%)",
            min_value = 10,
            max_value = 60,
            value     = 35,
            step      = 5
        )
    with b3:
        target_pct = st.slider(
            "% of high-risk customers to target",
            min_value = 10,
            max_value = 100,
            value     = 20,
            step      = 10
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Calculations using real numbers from model
    total_customers    = 7043
    churn_rate_val     = 0.265
    model_recall       = 0.76
    avg_monthly        = 74.44
    high_risk_pool     = int(total_customers * churn_rate_val * model_recall)
    targeted           = int(high_risk_pool * target_pct / 100)
    retained           = int(targeted * retention_rate / 100)
    revenue_saved      = retained * avg_monthly * 12
    campaign_cost      = targeted * offer_cost
    net_benefit        = revenue_saved - campaign_cost
    roi                = (net_benefit / campaign_cost) if campaign_cost > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("High-Risk Pool",     f"{high_risk_pool:,}",  "Flagged by model (76% recall)")
    r2.metric("Customers Targeted", f"{targeted:,}",        f"Top {target_pct}% by risk score")
    r3.metric("Estimated Retained", f"{retained:,}",        f"At {retention_rate}% success rate")
    r4.metric("Net Benefit",        f"${net_benefit:,.0f}", f"ROI: {roi:.1f}x")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        name        = "Campaign P&L",
        orientation = "v",
        measure     = ["absolute", "relative", "relative", "total"],
        x           = ["Revenue at Risk", "Campaign Cost", "Revenue Saved", "Net Benefit"],
        y           = [
            -(high_risk_pool * avg_monthly * 12),
            -campaign_cost,
            revenue_saved,
            net_benefit
        ],
        connector   = dict(line=dict(color='#e8e8e4')),
        decreasing  = dict(marker=dict(color='#ef4444')),
        increasing  = dict(marker=dict(color='#22c55e')),
        totals      = dict(marker=dict(color='#e8600a'))
    ))
    fig_wf.update_layout(
        title       = 'Campaign P&L Waterfall',
        plot_bgcolor= 'white', paper_bgcolor='white',
        title_font  = dict(size=14, color='#111110'),
        font        = dict(color='#737370', size=12),
        yaxis       = dict(
            tickprefix='$',
            tickformat=',.0f',
            gridcolor='#e8e8e4'
        ),
        margin      = dict(t=40, b=20),
        showlegend  = False
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Summary note
    st.markdown(f"""
    <div style='background:#fff8f5;border:1px solid #fde0cc;
                border-radius:8px;padding:18px 20px;
                font-size:13px;color:#737370;line-height:1.7;
                margin-top:8px'>
        <strong style='color:#e8600a'>How to read this:</strong>
        The model flags <strong style='color:#111110'>{high_risk_pool:,}</strong> high-risk
        customers from a base of 7,043. Targeting the top
        <strong style='color:#111110'>{target_pct}%</strong>
        ({targeted:,} customers) with a
        <strong style='color:#111110'>${offer_cost}</strong> retention offer,
        and assuming <strong style='color:#111110'>{retention_rate}%</strong>
        of them are retained, saves
        <strong style='color:#111110'>${revenue_saved:,.0f}</strong> in annual revenue
        at a campaign cost of <strong style='color:#111110'>${campaign_cost:,.0f}</strong>.
        Net benefit: <strong style='color:#111110'>${net_benefit:,.0f}</strong>
        ({roi:.1f}x ROI).
    </div>
    """, unsafe_allow_html=True)
