import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ─── 1. CONFIG & BRANDING ────────────────────────────────────────────────────
st.set_page_config(page_title="flyExclusive Pricing Tool", layout="wide")

BRAND_NAVY       = "#003B71"
BRAND_LIGHT_BLUE = "#3C97C6"
BRAND_LIGHT_GREY = "#CED9E5"
BRAND_DARK_GREY  = "#C3C5C4"
BRAND_GREEN      = "#2E7D32"
BRAND_AMBER      = "#F57C00"

def apply_brand_styling():
    st.markdown(f"""
        <style>
        h1, h2, h3 {{ color: {BRAND_NAVY} !important; font-family: sans-serif; }}
        div[data-testid="stMetricValue"] {{ color: {BRAND_NAVY}; }}
        div.stButton > button:first-child {{
            background-color: {BRAND_NAVY}; color: #FFFFFF;
            border: 1px solid {BRAND_NAVY}; border-radius: 4px;
        }}
        div.stButton > button:first-child:hover {{
            background-color: {BRAND_LIGHT_BLUE}; color: #FFFFFF; border-color: {BRAND_NAVY};
        }}
        </style>
    """, unsafe_allow_html=True)

apply_brand_styling()

# ─── 2. DATA LOADING ──────────────────────────────────────────────────────────
@st.cache_data
def load_data_fractional():
    try:
        try:
            df = pd.read_csv('raw data.csv')
        except FileNotFoundError:
            try:
                df = pd.read_csv('raw_data.csv')
            except FileNotFoundError:
                df = pd.read_csv('pricing training data.xlsx - raw data.csv')
        df.rename(columns={
            'Mangement Fee': 'Management Fee',
            'Amoritized management fee': 'Amortized Management Fee'
        }, inplace=True)
        for col in ['Company', 'type', 'Cabin type', 'Program Type', 'Aircraft Type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        for col in ['Hourly Rate', 'Day Rate', 'Purchase Price', 'Annual Membership Fees',
                    'Management Fee', 'Amortized Management Fee', 'Share %', 'Fractional Hours']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'Amortized Management Fee' not in df.columns:
            df['Amortized Management Fee'] = 0.0
        if 'Management Fee' not in df.columns:
            df['Management Fee'] = 0.0
        return df
    except FileNotFoundError:
        st.error("Could not find the fractional raw data CSV.")
        return pd.DataFrame()

@st.cache_data
def load_data_jc():
    try:
        try:
            df = pd.read_csv('jc raw data.csv')
        except FileNotFoundError:
            df = pd.read_csv('jc_raw_data.csv')
        df.rename(columns={'Type': 'Cabin type', 'Annual Membership fees': 'Annual Membership Fees'}, inplace=True)
        for col in ['Company', 'Cabin type', 'Program Name', 'Contract']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        for col in ['Fuel Surcharges', 'Mins', 'Daily', 'Hourly', 'Annual Membership Fees']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0
        return df
    except FileNotFoundError:
        st.error("Could not find 'jc raw data.csv'.")
        return pd.DataFrame()

# ─── DISCOUNT LOGIC HELPER ───────────────────────────────────────────────────
def get_discount_per_hr(row, avg_ft, is_fractional=True):
    if not is_fractional:
        return 0.0
    comp = str(row.get('Company', '')).lower()
    cab = str(row.get('Cabin type', '')).lower()
    ac = str(row.get('Aircraft Type', '')).lower()
    hr_rate = float(row.get('Hourly Rate', 0))
    disc = 0.0
    if 'flexjet' in comp:
        if 'phenom' in ac or 'light' in cab:
            if avg_ft >= 2.5: disc = 250
        elif 'praetor 500' in ac or 'mid' in cab:
            if avg_ft >= 2.5: disc = 500
        elif 'challenger' in ac or 'praetor 600' in ac or 'super mid' in cab:
            if avg_ft >= 2.5: disc = 1500
        elif 'g450' in ac or 'large' in cab:
            if 2.5 <= avg_ft <= 3.4: disc = 1000
            elif 3.5 <= avg_ft <= 4.4: disc = 1500
            elif avg_ft >= 4.5: disc = 2000
    elif 'netjets' in comp:
        if 'phenom' in ac or 'praetor' in ac or 'light' in cab:
            if 2.5 <= avg_ft <= 3.4: disc = 531
            elif 3.5 <= avg_ft <= 4.4: disc = 796
            elif avg_ft >= 4.5: disc = 1062
        elif 'ascend' in ac:
            if 2.5 <= avg_ft <= 3.4: disc = 557
            elif 3.5 <= avg_ft <= 4.4: disc = 836
            elif avg_ft >= 4.5: disc = 1114
        elif 'latitude' in ac or 'longitude' in ac or 'sovereign' in ac or 'challenger 3500' in ac or 'super mid' in cab:
            if 2.5 <= avg_ft <= 3.4: disc = 760
            elif 3.5 <= avg_ft <= 4.4: disc = 1140
            elif avg_ft >= 4.5: disc = 1520
        elif 'challenger 650' in ac or 'global' in ac or 'large' in cab or 'long range' in cab:
            if 2.5 <= avg_ft <= 3.4: disc = hr_rate * 0.20
            elif 3.5 <= avg_ft <= 4.4: disc = hr_rate * 0.30
            elif avg_ft >= 4.5: disc = hr_rate * 0.40
    return disc

# ─── 3. CALCULATION ENGINE ───────────────────────────────────────────────────
def calculate_master_table(df_frac, df_jc, avg_flight_time, annual_hours_slider,
                           depreciation_pct_new, depreciation_pct_used,
                           tax_rate, dep_term, cpi_rate):
    frac_res = df_frac.copy()
    frac_res['Program Category'] = 'Fractional'
    frac_res['Is FlyExclusive'] = frac_res['Company'].str.contains('flyexclusive', case=False, na=False)
    
    actual_trips = annual_hours_slider / avg_flight_time
    
    # Financials
    frac_res['Discount Per Hour'] = frac_res.apply(lambda r: get_discount_per_hr(r, avg_flight_time, True), axis=1)
    frac_res['Effective Hourly Rate Input'] = frac_res['Hourly Rate'] - frac_res['Discount Per Hour']
    frac_res['Annual Hourly Spend'] = annual_hours_slider * frac_res['Effective Hourly Rate Input']
    frac_res['Annual Day Spend']    = actual_trips * frac_res['Day Rate']
    frac_res['Annual Amort Mgmt']   = annual_hours_slider * frac_res['Amortized Management Fee']
    frac_res['Annual Fixed Spend']  = frac_res['Annual Membership Fees'] + frac_res['Management Fee']
    
    # Ownership Depreciation logic (Locked to Share Size, Burdened on usage)
    dep_pct = frac_res['type'].apply(lambda x: depreciation_pct_new if str(x).lower() == 'new' else depreciation_pct_used)
    frac_res['Total Annual Share Dep'] = (frac_res['Purchase Price'] * (dep_pct / 100)) / dep_term
    frac_res['Annual Depreciation'] = frac_res['Total Annual Share Dep']
    
    frac_res['Taxable Base'] = (frac_res['Annual Hourly Spend'] + frac_res['Annual Day Spend'] + frac_res['Annual Amort Mgmt'])
    frac_res['Annual Tax']   = np.where(frac_res['Program Type'] == 'JC', frac_res['Taxable Base'] * tax_rate, 0)
    frac_res['Annual Cost']  = (frac_res['Annual Hourly Spend'] + frac_res['Annual Day Spend'] + frac_res['Annual Amort Mgmt'] + frac_res['Annual Fixed Spend'] + frac_res['Annual Tax'] + frac_res['Annual Depreciation'])
    
    # Lifetime calculation
    lifetime_costs = []
    for _, row in frac_res.iterrows():
        total = 0
        ch, cd = row['Effective Hourly Rate Input'], row['Day Rate']
        for yr in range(int(dep_term)):
            if yr > 0:
                ch *= (1 + cpi_rate)
                cd *= (1 + cpi_rate)
            hs, ds, am = annual_hours_slider * ch, actual_trips * cd, annual_hours_slider * row['Amortized Management Fee']
            tax = (hs + ds + am) * tax_rate if row['Program Type'] == 'JC' else 0
            total += hs + ds + am + row['Annual Fixed Spend'] + tax + row['Annual Depreciation']
        lifetime_costs.append(total)
    frac_res['Total Lifetime Cost'] = lifetime_costs
    frac_res['Effective Hourly Rate'] = frac_res['Annual Cost'] / annual_hours_slider
    frac_res['Display Name'] = frac_res['Cabin type'] + ' ' + frac_res['type'].str.capitalize() + ' (' + frac_res['Aircraft Type'].fillna('Share') + ')'
    
    # Component Breakdowns
    frac_res['Variable Cost']         = (frac_res['Annual Hourly Spend'] + frac_res['Annual Day Spend'] + frac_res['Annual Amort Mgmt'])
    frac_res['Fixed Cost']            = frac_res['Annual Fixed Spend']
    frac_res['Tax Cost']              = frac_res['Annual Tax']
    frac_res['Depreciation Cost']     = frac_res['Annual Depreciation']
    
    # Hourly Breakdown Columns
    frac_res['Average Trip Cost (Hrly)'] = (frac_res['Variable Cost']) / annual_hours_slider
    frac_res['FET (Hrly)']               = frac_res['Annual Tax'] / annual_hours_slider
    frac_res['Membership & Fees (Hrly)'] = frac_res['Annual Fixed Spend'] / annual_hours_slider
    frac_res['Depreciation (Hrly)']      = frac_res['Total Annual Share Dep'] / annual_hours_slider
    
    # Jet Club Logic
    jc_res = df_jc.copy()
    if 'Aircraft Type' not in jc_res.columns: jc_res['Aircraft Type'] = jc_res['Cabin type']
    billed = jc_res['Mins'].apply(lambda x: max(x, avg_flight_time)) * actual_trips
    jc_res['Total Hourly Rate']    = jc_res['Hourly'] + jc_res['Fuel Surcharges']
    jc_res['Annual Hourly Spend']  = billed * jc_res['Total Hourly Rate']
    jc_res['Annual Day Spend']     = actual_trips * jc_res['Daily']
    jc_res['Annual Fixed Spend']   = jc_res['Annual Membership Fees']
    jc_res['Taxable Base']         = jc_res['Annual Hourly Spend'] + jc_res['Annual Day Spend']
    jc_res['Annual Tax']           = jc_res['Taxable Base'] * tax_rate
    jc_res['Annual Cost']          = (jc_res['Annual Hourly Spend'] + jc_res['Annual Day Spend'] + jc_res['Annual Fixed Spend']  + jc_res['Annual Tax'])
    
    lc_jc = []
    for _, row in jc_res.iterrows():
        total = 0
        ch, cf, cd = row['Hourly'], row['Fuel Surcharges'], row['Daily']
        for yr in range(int(dep_term)):
            if yr > 0:
                ch *= (1 + cpi_rate)
                cd *= (1 + cpi_rate)
            bh  = max(row['Mins'], avg_flight_time) * actual_trips
            hs, ds, tax = bh * (ch + cf), actual_trips * cd, (bh*(ch+cf) + actual_trips*cd) * tax_rate
            total += hs + ds + row['Annual Membership Fees'] + tax
        lc_jc.append(total)
    jc_res['Total Lifetime Cost']   = lc_jc
    jc_res['Effective Hourly Rate'] = jc_res['Annual Cost'] / annual_hours_slider
    jc_res['Program Category']      = 'Jet Club'
    jc_res['Display Name']          = jc_res['Program Name']
    jc_res['type']                  = 'Jet Card'
    jc_res['Variable Cost']         = jc_res['Annual Hourly Spend'] + jc_res['Annual Day Spend']
    jc_res['Fixed Cost']            = jc_res['Annual Fixed Spend']
    jc_res['Tax Cost']              = jc_res['Annual Tax']
    jc_res['Depreciation Cost']     = 0.0
    
    # Hourly Breakdown for Jet Club
    jc_res['Average Trip Cost (Hrly)'] = (jc_res['Annual Hourly Spend'] + jc_res['Annual Day Spend']) / annual_hours_slider
    jc_res['FET (Hrly)']               = jc_res['Annual Tax'] / annual_hours_slider
    jc_res['Membership & Fees (Hrly)'] = jc_res['Annual Fixed Spend'] / annual_hours_slider
    jc_res['Depreciation (Hrly)']      = 0.0
    
    common_cols = ['Display Name', 'Program Category', 'Company', 'Cabin type', 'Aircraft Type', 'type',
                   'Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost',
                   'Variable Cost', 'Fixed Cost', 'Tax Cost', 'Depreciation Cost', 
                   'Hourly Rate', 'Day Rate', 'Amortized Management Fee', 'Daily', 'Mins', 
                   'Total Hourly Rate', 'Fractional Hours', 'Average Trip Cost (Hrly)', 
                   'FET (Hrly)', 'Membership & Fees (Hrly)', 'Depreciation (Hrly)']
    
    for col in ['Daily', 'Mins', 'Total Hourly Rate', 'Fractional Hours']:
        if col not in frac_res.columns: frac_res[col] = 0.0
    for col in ['Hourly Rate', 'Day Rate', 'Amortized Management Fee', 'Depreciation Cost', 'Fractional Hours']:
        if col not in jc_res.columns: jc_res[col] = 0.0
        
    return pd.concat([frac_res[common_cols], jc_res[common_cols]], ignore_index=True)

def is_fe(company): return 'flyexclusive' in str(company).lower()
def get_target_tier(usage, tiers):
    v = [t for t in tiers if t >= usage]
    return min(v) if v else (max(tiers) if tiers else 50)

# ─── 4. DATA INIT ─────────────────────────────────────────────────────────────
df_frac, df_jc = load_data_fractional(), load_data_jc()
if df_frac.empty or df_jc.empty: st.stop()
all_cabins     = sorted(set(df_frac['Cabin type'].unique()) | set(df_jc['Cabin type'].unique()))
all_companies  = sorted(set(df_frac['Company'].unique())    | set(df_jc['Company'].unique()))

# Initialize Session State for Manual Entries
if 'custom_frac_forecast' not in st.session_state:
    st.session_state['custom_frac_forecast'] = pd.DataFrame(columns=['Aircraft Type', 'Cabin type', 'type', 'Purchase Price', 'Hourly Rate', 'Day Rate', 'Management Fee', 'Amortized Management Fee', 'Annual Membership Fees', 'Fractional Hours'])
if 'custom_jc_forecast' not in st.session_state:
    st.session_state['custom_jc_forecast'] = pd.DataFrame(columns=['Program Name', 'Cabin type', 'Hourly', 'Daily', 'Mins', 'Fuel Surcharges', 'Annual Membership Fees'])

# ─── 5. SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://22283886.fs1.hubspotusercontent-na1.net/hubfs/22283886/Caravan%20Drip%20Images/FLY_MasterLogo_CMYK_PMS541.png", use_container_width=True)
    st.markdown("---")
    st.header("1. Usage Inputs")
    annual_hours = st.slider("Annual Usage (Hours)", 10, 500, 50, 5)
    calc_basis   = st.radio("Trip Calculator", ["Average Flight Time", "Total Trips per Year"])
    if calc_basis == "Average Flight Time":
        avg_flight_time = st.number_input("Average Flight Time (Hours)", 0.5, value=2.0, step=0.1)
    else:
        total_trips     = st.number_input("Total Trips per Year", 1, value=25, step=1)
        avg_flight_time = annual_hours / total_trips
    
    st.header("2. Financial Inputs")
    depreciation_pct_new, depreciation_pct_used = st.slider("Depreciation - New (%)", 0, 100, 60), st.slider("Depreciation - Used (%)", 0, 100, 50)
    with st.expander("Logic Parameters"):
        tax_rate, dep_term, cpi_rate = st.number_input("FET Tax Rate", value=0.075, format="%.3f"), st.number_input("Term (Years)", value=5), st.number_input("CPI (Annual %)", value=0.03, format="%.3f")

    st.markdown("---")
    st.header("3. Forecast Scenarios")
    with st.expander("Add Fractional Program"):
        with st.form("frac_form"):
            ff_name       = st.text_input("Name", "Forecast Frax")
            ff_cabin      = st.selectbox("Cabin", all_cabins, key='ff_cab')
            ff_hours      = st.number_input("Share Size", 50)
            ff_price      = st.number_input("Purchase Price", 500000)
            ff_hourly     = st.number_input("Hourly Rate", 3500)
            col3, col4    = st.columns(2)
            ff_day        = col3.number_input("Day Rate ($)", min_value=0, value=0, step=100)
            ff_mgmt_amort = col4.number_input("Amortized Mgmt ($/hr)", min_value=0.0, value=0.0, step=10.0)
            ff_annual_mem = st.number_input("Annual Membership Fee ($)", min_value=0, value=0, step=1000)
            if st.form_submit_button("Add to Analysis"):
                new_row = pd.DataFrame([{
                    'Aircraft Type': ff_name, 'Cabin type': ff_cabin, 'type': 'New',
                    'Program Type': 'Frax', 'Fractional Hours': ff_hours,
                    'Purchase Price': ff_price, 'Hourly Rate': ff_hourly,
                    'Day Rate': ff_day, 'Management Fee': 0,
                    'Amortized Management Fee': ff_mgmt_amort,
                    'Annual Membership Fees': ff_annual_mem,
                    'Company': 'flyExclusive (Forecast)'
                }])
                st.session_state['custom_frac_forecast'] = pd.concat([st.session_state['custom_frac_forecast'], new_row], ignore_index=True)
                st.rerun()

    with st.expander("Add Jet Club Program"):
        with st.form("jc_form"):
            jf_name   = st.text_input("Name", "Forecast JC")
            jf_cabin  = st.selectbox("Cabin", all_cabins, key='jf_cab')
            jf_hourly = st.number_input("Hourly Rate", 4000)
            jf_daily  = st.number_input("Daily Rate", 0)
            jf_mins   = st.number_input("Daily Min", 1.2)
            if st.form_submit_button("Add to Analysis"):
                new_row = pd.DataFrame([{
                    'Program Name': jf_name, 'Cabin type': jf_cabin,
                    'Hourly': jf_hourly, 'Daily': jf_daily, 'Mins': jf_mins,
                    'Fuel Surcharges': 0, 'Annual Membership Fees': 0,
                    'Company': 'flyExclusive (Forecast)'
                }])
                st.session_state['custom_jc_forecast'] = pd.concat([st.session_state['custom_jc_forecast'], new_row], ignore_index=True)
                st.rerun()

    if st.button("Clear Manual Forecasts"):
        st.session_state['custom_frac_forecast'] = st.session_state['custom_frac_forecast'].iloc[0:0]
        st.session_state['custom_jc_forecast']   = st.session_state['custom_jc_forecast'].iloc[0:0]
        st.rerun()

# ─── 6. MERGE & FILTER ───────────────────────────────────────────────────────
df_frac_combined = pd.concat([df_frac, st.session_state['custom_frac_forecast']], ignore_index=True)
df_jc_combined = pd.concat([df_jc, st.session_state['custom_jc_forecast']], ignore_index=True)

st.title("flyExclusive Pricing Tool")
all_aircraft = sorted(set(df_frac_combined['Aircraft Type'].fillna('Generic').unique()) | set(df_jc_combined['Cabin type'].unique()))
c1, c2, c3, c4 = st.columns(4)
sel_cabin, sel_company, sel_type, sel_aircraft = c1.multiselect("Filter Cabin Type", all_cabins), c2.multiselect("Filter Company", all_companies), c3.multiselect("Filter Program Category", ["Fractional", "Jet Club"]), c4.multiselect("Filter Aircraft Type", all_aircraft)
target_tier = get_target_tier(annual_hours, sorted([t for t in df_frac_combined['Fractional Hours'].unique() if t > 0]))

if sel_cabin: df_frac_combined, df_jc_combined = df_frac_combined[df_frac_combined['Cabin type'].isin(sel_cabin)], df_jc_combined[df_jc_combined['Cabin type'].isin(sel_cabin)]
if sel_company: df_frac_combined, df_jc_combined = df_frac_combined[df_frac_combined['Company'].isin(sel_company)], df_jc_combined[df_jc_combined['Company'].isin(sel_company)]

df_frac_combined = df_frac_combined[(df_frac_combined['Fractional Hours'] == target_tier) | (df_frac_combined['Fractional Hours'] == 0)]
master_df = calculate_master_table(df_frac_combined, df_jc_combined, avg_flight_time, annual_hours, depreciation_pct_new, depreciation_pct_used, tax_rate, dep_term, cpi_rate)
if sel_type: master_df = master_df[master_df['Program Category'].isin(sel_type)]
if sel_aircraft: master_df = master_df[master_df['Aircraft Type'].isin(sel_aircraft)]

# ─── 7. MARKET BENCHMARK KPI BANNER ───────────────────────────────────────────
if not master_df.empty:
    fe_df = master_df[master_df['Company'].str.contains('flyexclusive', case=False, na=False)]
    comp_df = master_df[~master_df['Company'].str.contains('flyexclusive', case=False, na=False)]

    if not fe_df.empty and not comp_df.empty:
        st.subheader("Market Benchmark: flyExclusive vs. Competition")
        fe_avg   = fe_df[['Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost']].mean()
        comp_avg = comp_df[['Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost']].mean()
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            diff_ehr = comp_avg['Effective Hourly Rate'] - fe_avg['Effective Hourly Rate']
            st.metric("Avg Effective Hourly", f"${fe_avg['Effective Hourly Rate']:,.0f}", f"${diff_ehr:,.0f} vs Market", delta_color="normal")
        with kpi2:
            diff_ann = comp_avg['Annual Cost'] - fe_avg['Annual Cost']
            st.metric("Avg Annual Cost", f"${fe_avg['Annual Cost']:,.0f}", f"${diff_ann:,.0f} vs Market", delta_color="normal")
        with kpi3:
            diff_life = comp_avg['Total Lifetime Cost'] - fe_avg['Total Lifetime Cost']
            st.metric(f"Avg {int(dep_term)}-Year Cost", f"${fe_avg['Total Lifetime Cost']:,.0f}", f"${diff_life:,.0f} vs Market", delta_color="normal")
        st.markdown("---")

# ─── 8. DASHBOARD TABS ────────────────────────────────────────────────────────
if not master_df.empty:
    master_df['Is flyExclusive'] = master_df['Company'].apply(is_fe)
    grouped = master_df.groupby(['Company', 'Cabin type', 'Program Category', 'Is flyExclusive'], as_index=False)[['Annual Cost', 'Total Lifetime Cost', 'Effective Hourly Rate', 'Variable Cost', 'Fixed Cost', 'Tax Cost', 'Depreciation Cost']].mean()
    grouped['Bar Label'] = grouped['Company'] + ' — ' + grouped['Cabin type'] + ' (' + grouped['Program Category'] + ')'
    
    t1, t2, t3 = st.tabs(["Cost Breakdown", f"{int(dep_term)}-Year Lifetime Cost", "Effective Hourly (Heatmap)"])
    
    with t1:
        df_stack = grouped.sort_values('Annual Cost', ascending=False)
        fig_stack = go.Figure()
        for c, color, lbl in [('Variable Cost', '#1565C0', 'Variable'), ('Fixed Cost', BRAND_LIGHT_BLUE, 'Fixed'), ('Tax Cost', BRAND_AMBER, 'FET Tax'), ('Depreciation Cost', BRAND_DARK_GREY, 'Depreciation')]:
            fig_stack.add_trace(go.Bar(name=lbl, y=df_stack['Bar Label'], x=df_stack[c], marker_color=color, orientation='h'))
        
        fig_stack.add_trace(go.Scatter(
            y=df_stack['Bar Label'],
            x=df_stack['Annual Cost'],
            mode='text',
            text=df_stack['Annual Cost'].apply(lambda x: f" ${x:,.0f}"),
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip'
        ))

        fig_stack.update_layout(barmode='stack', height=max(500, len(df_stack)*35), yaxis=dict(autorange='reversed'), xaxis_tickformat='$,.0f', margin=dict(r=100))
        st.plotly_chart(fig_stack, use_container_width=True)

    with t2:
        df_life = grouped.sort_values('Total Lifetime Cost', ascending=False)
        fig_life = go.Figure()
        for _, row in df_life.iterrows():
            color = BRAND_NAVY if row['Is flyExclusive'] else BRAND_LIGHT_BLUE
            fig_life.add_trace(go.Bar(y=[row['Bar Label']], x=[row['Total Lifetime Cost']], marker_color=color, orientation='h'))
        fig_life.update_layout(title="Total Ownership Cost over Term", xaxis_tickformat='$,.0f', height=max(500, len(df_life)*35), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_life, use_container_width=True)

    cabin_caps, trip_times = {'Light': 3.5, 'Mid': 4.0, 'Super Mid': 6.0, 'Heavy': 6.0}, [round(x, 1) for x in np.arange(1.0, 8.1, 0.1)]
    sens_rows = []
    for _, r in master_df.iterrows():
        sh = r.get('Fractional Hours', 0) if r.get('Fractional Hours', 0) > 0 else 50
        for tt in trip_times:
            if tt > cabin_caps.get(r['Cabin type'], 8.0): continue
            trps = sh / tt
            if r['Program Category'] == 'Fractional':
                ds = get_discount_per_hr(r, tt)
                tot = sh*(r['Hourly Rate']-ds) + trps*r['Day Rate'] + r['Fixed Cost'] + r['Depreciation Cost'] + r['Tax Cost']
            else:
                btt = max(r['Mins'], tt)
                hs, ds = trps*btt*r['Total Hourly Rate'], trps*r['Daily']
                tot = hs + ds + r['Fixed Cost'] + ((hs+ds)*tax_rate)
            sens_rows.append({'Label': f"{r['Company']} - {r['Cabin type']} ({r['Program Category']})", 'Time': tt, 'EHR': round(tot/sh)})
    sens_df = pd.DataFrame(sens_rows)

    with t3:
        if not sens_df.empty:
            pivot = sens_df.pivot_table(index='Label', columns='Time', values='EHR')
            fig_heat = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, text=pivot.values, texttemplate="$%{z:,.0f}", colorscale='Blues', colorbar=dict(tickformat='$,.0f')))
            fig_heat.update_layout(title="Effective Hourly Rates by Flight Time", height=max(400, len(pivot)*35), yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Unified Program Comparison")
    sorted_df = master_df.sort_values(['Is flyExclusive', 'Effective Hourly Rate'], ascending=[False, True]).copy()
    sorted_df['Aircraft Type'] = np.where(sorted_df['type'] == 'Jet Card', '—', sorted_df['Aircraft Type'])
    currency_cols = ['Average Trip Cost (Hrly)', 'FET (Hrly)', 'Depreciation (Hrly)', 'Membership & Fees (Hrly)', 'Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost']
    render_df = sorted_df[['Company', 'Display Name', 'Aircraft Type', 'type'] + currency_cols]
    st.dataframe(render_df.style.apply(lambda r: [f'background-color: {BRAND_LIGHT_GREY}; font-weight: bold;' if is_fe(r['Company']) else '']*len(r), axis=1).format({c: '${:,.0f}' for c in currency_cols}), use_container_width=True)

    with st.expander("Configure Report"):
        rep_title = st.text_input("Report Title", "Competitive Analysis Report")
        if st.button("Generate Report"):
            rep_tbl = render_df.copy()
            for c in currency_cols: rep_tbl[c] = rep_tbl[c].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else '—')
            rep_tbl_html = rep_tbl.to_html(index=False, classes='summary-table', border=0)
            
            fig_stack_rep = go.Figure()
            for c, color, lbl in [('Variable Cost', '#1565C0', 'Variable'), ('Fixed Cost', BRAND_LIGHT_BLUE, 'Fixed'), ('Tax Cost', BRAND_AMBER, 'FET Tax'), ('Depreciation Cost', BRAND_DARK_GREY, 'Depreciation')]:
                fig_stack_rep.add_trace(go.Bar(name=lbl, y=df_stack['Bar Label'], x=df_stack[c], marker_color=color, orientation='h'))
            
            fig_stack_rep.add_trace(go.Scatter(y=df_stack['Bar Label'], x=df_stack['Annual Cost'], mode='text', text=df_stack['Annual Cost'].apply(lambda x: f" ${x:,.0f}"), textposition='middle right', showlegend=False))
            fig_stack_rep.update_layout(barmode='stack', height=max(400, len(df_stack)*30), yaxis=dict(autorange='reversed'), xaxis_tickformat='$,.0f')
            
            pivot_rep = sens_df.pivot_table(index='Label', columns='Time', values='EHR')
            fig_heat_rep = go.Figure(go.Heatmap(z=pivot_rep.values, x=pivot_rep.columns, y=pivot_rep.index, colorscale='Blues', text=pivot_rep.values, texttemplate="$%{z:,.0f}"))
            fig_heat_rep.update_layout(title="Effective Hourly Rates", height=max(400, len(pivot_rep)*30), yaxis=dict(autorange='reversed'))
            
            html = f"""<!DOCTYPE html><html><head><script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script><style>
                body {{ font-family: sans-serif; padding: 40px; background: #f4f6f9; }}
                .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
                h1, h2 {{ color: {BRAND_NAVY}; }}
                .summary-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
                .summary-table th {{ background: {BRAND_NAVY}; color: white; padding: 8px; text-align: left; }}
                .summary-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
            </style></head><body>
                <h1>{rep_title}</h1>
                <div class="card"><h2>Comparison Table</h2>{rep_tbl_html}</div>
                <div class="card"><h2>Annual Cost Breakdown</h2>{pio.to_html(fig_stack_rep, full_html=False)}</div>
                <div class="card"><h2>Effective Hourly Rates</h2>{pio.to_html(fig_heat_rep, full_html=False)}</div>
            </body></html>"""
            st.download_button("Download Analysis Report", html.encode('utf-8'), "aviation_report.html", "text/html")
            st.success("Report Ready!")