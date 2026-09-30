
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Afficionado Coffee Roasters | Analytics Portfolio",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
MODEL = ROOT / "models" / "demand_model.joblib"

COFFEE = "#6f3f22"
COFFEE_DARK = "#4b2818"
LATTE = "#d7b48c"
CREAM = "#f6eadb"
ESPRESSO = "#2d1b12"
MUTED = "#7c6759"

st.markdown("""
<style>
:root { --coffee:#6f3f22; --dark:#4b2818; --cream:#f6eadb; --latte:#d7b48c; }
.stApp {
    background: linear-gradient(135deg, #fbf3e8 0%, #ead6bd 52%, #f7eee3 100%);
    color: #2d1b12;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4b2818 0%, #6f3f22 100%);
}
[data-testid="stSidebar"] * { color: #fffaf4 !important; }
.hero {
    padding: 2.1rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(75,40,24,.97), rgba(111,63,34,.91));
    color: #fffaf4;
    box-shadow: 0 12px 35px rgba(60,30,15,.22);
    margin-bottom: 1.4rem;
}
.hero h1 { font-size: 2.8rem; margin: 0 0 .35rem 0; }
.hero p { font-size: 1.05rem; margin: .2rem 0; opacity:.92; }
.section {
    padding: .6rem 0 .2rem 0;
}
.kpi {
    background: rgba(255,250,244,.72);
    border: 1px solid rgba(111,63,34,.28);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 6px 20px rgba(80,45,25,.08);
}
.kpi .label { color:#7c6759; font-size:.86rem; }
.kpi .value { color:#6f3f22; font-size:1.65rem; font-weight:750; }
.insight {
    background: rgba(255,250,244,.70);
    border-left: 5px solid #6f3f22;
    border-radius: 10px;
    padding: .9rem 1rem;
    margin: .4rem 0;
}
.small-note { color:#7c6759; font-size:.86rem; }
.footer {
    margin-top: 2.5rem; padding: 1.2rem; text-align:center;
    color:#7c6759; font-size:.85rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "transactions_clean.csv")
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

df = load_data()

# Derived summaries
store_summary = (df.groupby(["store_id","store_location"], as_index=False)
                   .agg(transactions=("transaction_id","nunique"),
                        quantity=("transaction_qty","sum"),
                        revenue=("revenue","sum"),
                        avg_transaction_value=("revenue","mean"))
                   .sort_values("revenue", ascending=False))
cat_summary = (df.groupby("product_category", as_index=False)
                 .agg(transactions=("transaction_id","nunique"),
                      quantity=("transaction_qty","sum"),
                      revenue=("revenue","sum"))
                 .sort_values("revenue", ascending=False))
hourly = (df.groupby(["store_location","hour","daypart"], as_index=False)
            .agg(transactions=("transaction_id","nunique"),
                 quantity=("transaction_qty","sum"),
                 revenue=("revenue","sum")))
hourly_total = (hourly.groupby("hour", as_index=False)
                      .agg(quantity=("quantity","sum"),
                           revenue=("revenue","sum"),
                           transactions=("transactions","sum")))
thresholds = hourly.groupby("store_location")["quantity"].transform(lambda s: s.quantile(.75))
hourly["is_peak_hour"] = hourly["quantity"] >= thresholds
peak = hourly[hourly["is_peak_hour"]].sort_values(["store_location","quantity"], ascending=[True,False])

revenue = df["revenue"].sum()
quantity = df["transaction_qty"].sum()
transactions = df["transaction_id"].nunique()
atv = df["revenue"].mean()

# Sidebar
with st.sidebar:
    st.markdown("## ☕ Afficionado")
    st.caption("Coffee analytics portfolio")
    page = st.radio(
        "Explore project",
        ["Home", "Executive Overview", "Peak Demand & Operations",
         "Product & Store Analysis", "Forecasting & Demand Planning",
         "EDA & Data Explorer", "Power BI Reference"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Project stack")
    st.markdown("**Python** · Pandas · Scikit-learn")
    st.markdown("**Visualization** · Plotly")
    st.markdown("**Dashboard** · Power BI")
    st.markdown("**Deployment** · Streamlit")
    st.divider()
    st.caption("Source: supplied Afficionado Coffee Roasters notebook, cleaned transaction data, PBIX and EDA report.")

def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

def chart_layout(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=20,r=20,t=55,b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,244,.20)",
        font=dict(color=ESPRESSO),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

if page == "Home":
    hero("Afficionado Coffee Roasters", "Data-Driven Forecasting & Peak Demand Prediction — interactive portfolio")
    kpi_row([
        ("Total Revenue", f"${revenue:,.2f}"),
        ("Total Quantity", f"{quantity:,.0f}"),
        ("Transactions", f"{transactions:,.0f}"),
        ("Avg. Transaction Value", f"${atv:,.2f}"),
    ])
    st.write("")
    left, right = st.columns([1.15, .85])
    with left:
        if (ASSETS/"home.png").exists():
            st.image(str(ASSETS/"home.png"), use_container_width=True)
    with right:
        st.markdown("### Project story")
        st.markdown("""
        This portfolio turns the supplied EDA, cleaned dataset, machine-learning notebook
        and Power BI report into one web experience. It preserves the project's original
        analytical structure while making the key findings interactive.
        """)
        st.markdown('<div class="insight"><b>Morning demand is the key operating window.</b><br>Across the three stores, the supplied analysis repeatedly identifies 07:00–10:00 as a high-demand period.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><b>Coffee + Tea are the core revenue categories.</b><br>Together they contribute 66.7% of recorded revenue in the supplied category summary.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><b>Store revenue is balanced.</b><br>Each location contributes roughly one-third of total revenue, while basket value differs by store.</div>', unsafe_allow_html=True)
    st.markdown("### Portfolio deliverables")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("EDA", "Complete")
    c2.metric("Peak analysis", "07:00–10:00")
    c3.metric("ML baseline", "MAE 0.4365")
    c4.metric("Dashboard", "5-page PBIX")
    st.info("Important scope note: the supplied source has year + time-of-day but no true calendar transaction date. The notebook therefore creates 2025-01-01 as the transaction date. This app presents the model as an intraday prediction baseline, not as a validated multi-day/seasonal forecast.")

elif page == "Executive Overview":
    hero("Executive Overview", "A management view of revenue, volume, stores and category mix.")
    kpi_row([
        ("Revenue", f"${revenue/1000:,.2f}K"),
        ("Quantity", f"{quantity/1000:,.0f}K"),
        ("Transactions", f"{transactions/1000:,.0f}K"),
        ("ATV", f"${atv:,.2f}"),
    ])
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(store_summary, x="revenue", y="store_location", orientation="h",
                   title="Revenue by Store Location", text_auto=".2s")
        fig.update_traces(marker_color=COFFEE)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with c2:
        fig=px.bar(cat_summary, x="revenue", y="product_category", orientation="h",
                   title="Revenue by Product Category", text_auto=".2s")
        fig.update_traces(marker_color=LATTE)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.markdown("### Store performance")
    display=store_summary.copy()
    display["revenue"]=display["revenue"].map(lambda x:f"${x:,.2f}")
    display["avg_transaction_value"]=display["avg_transaction_value"].map(lambda x:f"${x:,.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.markdown("### Category mix")
    cat_display=cat_summary.copy()
    cat_display["revenue_share"]=cat_display["revenue"]/revenue
    cat_display["revenue_share"]=cat_display["revenue_share"].map(lambda x:f"{x:.1%}")
    cat_display["revenue"]=cat_display["revenue"].map(lambda x:f"${x:,.2f}")
    st.dataframe(cat_display, use_container_width=True, hide_index=True)

elif page == "Peak Demand & Operations":
    hero("Peak Demand & Operations", "Find the busiest hours and translate them into operational actions.")
    store_filter=st.selectbox("Store", ["All"] + sorted(df["store_location"].unique().tolist()))
    plot_df=hourly_total if store_filter=="All" else hourly[hourly.store_location==store_filter].groupby("hour",as_index=False).agg(quantity=("quantity","sum"),revenue=("revenue","sum"),transactions=("transactions","sum"))
    fig=px.line(plot_df, x="hour", y="quantity", markers=True, title="Total Quantity by Hour")
    fig.update_traces(line_color=COFFEE, marker_color=COFFEE)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(chart_layout(fig,470), use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        hm=hourly.pivot_table(index="store_location",columns="hour",values="quantity",aggfunc="sum",fill_value=0)
        fig=px.imshow(hm, text_auto=True, aspect="auto", title="Demand Heatmap — Store vs Hour",
                      color_continuous_scale=[[0,"#f6eadb"],[.5,"#c28c63"],[1,"#6f3f22"]])
        st.plotly_chart(chart_layout(fig,400), use_container_width=True)
    with c2:
        p=peak.copy()
        if store_filter!="All": p=p[p.store_location==store_filter]
        p=p.sort_values("quantity",ascending=False).head(12)
        fig=px.bar(p, x="quantity", y="store_location", color="hour", orientation="h",
                   title="Peak-Hour Quantity")
        st.plotly_chart(chart_layout(fig,400), use_container_width=True)
    st.markdown("### Peak-hour table")
    p=peak.copy()
    if store_filter!="All": p=p[p.store_location==store_filter]
    p["hour"]=p["hour"].map(lambda x:f"{int(x):02d}:00")
    p["revenue"]=p["revenue"].map(lambda x:f"${x:,.2f}")
    st.dataframe(p, use_container_width=True, hide_index=True)
    st.markdown("### Operational interpretation")
    st.markdown('<div class="insight"><b>Staffing:</b> align stronger front-of-house coverage with the identified morning rush.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight"><b>Replenishment:</b> pre-position core beverage inventory before the peak window and monitor stockouts.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight"><b>Store-specific thresholds:</b> the notebook uses a store-level 75th-percentile hourly quantity rule to define peak hours.</div>', unsafe_allow_html=True)

elif page == "Product & Store Analysis":
    hero("Product & Store Analysis", "Explore category concentration, store performance and basket value.")
    c1,c2=st.columns(2)
    with c1:
        fig=px.treemap(cat_summary, path=["product_category"], values="revenue",
                       color="revenue", color_continuous_scale=[[0,"#f6eadb"],[1,"#6f3f22"]],
                       title="Revenue Mix by Product Category")
        st.plotly_chart(chart_layout(fig,470), use_container_width=True)
    with c2:
        fig=px.scatter(cat_summary, x="quantity", y="revenue", size="transactions",
                       hover_name="product_category", title="Quantity vs Revenue by Category")
        fig.update_traces(marker=dict(color=COFFEE))
        st.plotly_chart(chart_layout(fig,470), use_container_width=True)
    st.markdown("### Store comparison")
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=store_summary["store_location"], y=store_summary["revenue"], marker_color=COFFEE))
    fig.add_trace(go.Bar(name="Quantity", x=store_summary["store_location"], y=store_summary["quantity"], marker_color=LATTE, yaxis="y2"))
    fig.update_layout(title="Revenue and Quantity by Store", yaxis=dict(title="Revenue"), yaxis2=dict(title="Quantity",overlaying="y",side="right"), barmode="group")
    st.plotly_chart(chart_layout(fig,430), use_container_width=True)
    st.markdown("### Category detail")
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

elif page == "Forecasting & Demand Planning":
    hero("Forecasting & Demand Planning", "Interactive intraday quantity prediction using the supplied notebook model.")
    st.warning("Model scope: this is the supplied HistGradientBoostingRegressor baseline. Because the source lacks a true calendar date and the notebook split is ordered by hour/minute rather than future dates, MAE/RMSE should not be interpreted as validated future or seasonal forecast accuracy.")
    kpi_row([("MAE", "0.4365"), ("RMSE", "0.4899"), ("Train/Test", "80% / 20%"), ("Model", "HistGradientBoosting")])
    st.markdown("### Demand predictor")
    c1,c2,c3=st.columns(3)
    with c1:
        store=st.selectbox("Store location", sorted(df.store_location.unique()))
        category=st.selectbox("Product category", sorted(df.product_category.unique()))
    with c2:
        ptypes=sorted(df.loc[df.product_category==category,"product_type"].dropna().unique())
        ptype=st.selectbox("Product type", ptypes)
        unit_price=st.number_input("Unit price", min_value=0.0, value=float(df.loc[df.product_category==category,"unit_price"].median()), step=0.1)
    with c3:
        hour=st.slider("Hour", 6, 20, 10)
        minute=st.slider("Minute", 0, 59, 0)
    if st.button("Predict quantity", type="primary"):
        model=load_model()
        inp=pd.DataFrame([{"unit_price":unit_price,"hour":hour,"minute":minute,"store_location":store,"product_category":category,"product_type":ptype}])
        pred=float(model.predict(inp)[0])
        st.success(f"Predicted transaction quantity: **{pred:.2f} units**")
        st.caption("Prediction is a model estimate under the same feature definition used in the supplied notebook.")
    st.markdown("### Recommended next modelling step")
    st.markdown("""
    The supplied EDA recommends obtaining a genuine transaction date, aggregating demand by date × store × product/category
    (or hour × store), creating lag/rolling/calendar/promotion features, and evaluating with chronological backtesting.
    """)
    fig=px.line(hourly_total, x="hour", y=["quantity","revenue"], title="Intraday demand and revenue profile")
    st.plotly_chart(chart_layout(fig,430), use_container_width=True)

elif page == "EDA & Data Explorer":
    hero("EDA & Data Explorer", "Filter the cleaned transaction-level dataset and inspect the underlying records.")
    c1,c2,c3=st.columns(3)
    with c1: store=st.multiselect("Store location", sorted(df.store_location.unique()), default=sorted(df.store_location.unique()))
    with c2: cats=st.multiselect("Product category", sorted(df.product_category.unique()), default=sorted(df.product_category.unique()))
    with c3: dayparts=st.multiselect("Daypart", sorted(df.daypart.dropna().unique()), default=sorted(df.daypart.dropna().unique()))
    q=df[df.store_location.isin(store)&df.product_category.isin(cats)&df.daypart.isin(dayparts)].copy()
    kpi_row([("Filtered revenue",f"${q.revenue.sum():,.2f}"),("Filtered quantity",f"{q.transaction_qty.sum():,.0f}"),("Rows",f"{len(q):,.0f}"),("Filtered ATV",f"${q.revenue.mean():,.2f}")])
    st.write("")
    h=q.groupby("hour",as_index=False).agg(quantity=("transaction_qty","sum"),revenue=("revenue","sum"))
    fig=px.bar(h,x="hour",y="quantity",title="Filtered Quantity by Hour")
    fig.update_traces(marker_color=COFFEE)
    st.plotly_chart(chart_layout(fig,380),use_container_width=True)
    st.dataframe(q.sort_values("transaction_datetime").head(1000),use_container_width=True,hide_index=True)
    st.caption("Showing the first 1,000 matching rows for browser performance.")

elif page == "Power BI Reference":
    hero("Power BI Reference", "The original five-page dashboard is preserved as a visual reference alongside this Streamlit rebuild.")
    st.info("The PBIX itself cannot be rendered natively inside Streamlit. This portfolio app recreates the analytical experience in Python/Plotly and keeps the supplied PBIX available as a project artifact.")
    tabs=st.tabs(["Home","Executive Overview","Peak Demand","Product & Store","Forecasting"])
    names=["home.png","executive_overview.png","peak_demand.png","product_store.png","forecasting.png"]
    labels=["Home","Executive Overview","Peak Demand","Product & Store","Forecasting"]
    for tab,name,label in zip(tabs,names,labels):
        with tab:
            st.image(str(ASSETS/name), caption=f"Original Power BI — {label}", use_container_width=True)
    st.markdown("### Included project artifacts")
    st.markdown("""
    - Clean transaction dataset
    - Aggregated store/category/peak-hour CSVs
    - Original EDA + forecasting notebook
    - EDA & Business Insights report
    - Original Power BI PBIX
    - Pre-trained scikit-learn demand model
    """)

st.markdown('<div class="footer">Afficionado Coffee Roasters · Data Analytics Portfolio · Built with Streamlit, Pandas, Plotly and Scikit-learn</div>', unsafe_allow_html=True)
