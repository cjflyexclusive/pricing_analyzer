import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Define a simple password check
def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state.password_correct = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    # Show input for password
    st.text_input("Please enter the password to access the analysis:", type="password", key="password", on_change=password_entered)
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("Password incorrect")
    return False

# Stop the app if password is not correct
if not check_password():
    st.stop()

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Private Aviation Cost Analyzer", layout="wide", page_icon="✈️")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data_fractional():
    try:
        try:
            df = pd.read_csv('raw data.csv')
        except FileNotFoundError:
            df = pd.read_csv('pricing training data.xlsx - raw data.csv')
            
        cols_to_clean = ['Company', 'type', 'Cabin type', 'Program Type', 'Aircraft Type']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        num_cols = ['Hourly Rate', 'Day Rate', 'Purchase Price', 
                    'Annual Membership Fees', 'Mangement Fee', 'Amoritized management fee',
                    'Share %', 'Fractional Hours']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        if 'Amoritized management fee' not in df.columns:
            df['Amoritized management fee'] = 0.0
            
        return df
    except FileNotFoundError:
        st.error("Could not find the fractional raw data CSV.")
        return pd.DataFrame()

@st.cache_data
def load_data_jc():
    try:
        df = pd.read_csv('jc raw data.csv')
        df.rename(columns={'Type': 'Cabin type', 'Annual Membership fees': 'Annual Membership Fees'}, inplace=True)
        
        cols_to_clean = ['Company', 'Cabin type', 'Program Name', 'Contract']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                
        num_cols = ['Fuel Surcharges', 'Mins', 'Daily', 'Hourly', 'Annual Membership Fees']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0
                
        return df
    except FileNotFoundError:
        st.error("Could not find 'jc raw data.csv'. Please ensure it is in the same folder.")
        return pd.DataFrame()

# --- 3. LOGIC ENGINES ---
def calculate_costs_fractional(df, avg_flight_time, annual_usage, depreciation_pct_new, depreciation_pct_used, 
                               tax_rate, depreciation_term_years, cpi_rate):
    results = df.copy()
    
    # Base Year Calculations
    results['Trip Cost'] = ((avg_flight_time * results['Hourly Rate']) + results['Day Rate'] + 
                            (avg_flight_time * results['Amoritized management fee']))
    
    results['Trip Cost (FET)'] = results.apply(
        lambda x: x['Trip Cost'] * (1 + tax_rate) if x['Program Type'] == 'JC' else x['Trip Cost'], axis=1
    )
    
    total_term_hours = annual_usage * depreciation_term_years
    if total_term_hours > 0:
        dep_pct_series = results['type'].apply(
            lambda x: depreciation_pct_new if str(x).lower() == 'new' else depreciation_pct_used
        )
        results['Depreciation Hourly'] = (results['Purchase Price'] * (dep_pct_series / 100)) / total_term_hours
    else:
        results['Depreciation Hourly'] = 0.0

    results['Total Trip Cost'] = results['Trip Cost (FET)'] + (results['Depreciation Hourly'] * avg_flight_time)
    results['Effective Hourly Rate'] = np.where(avg_flight_time > 0, results['Total Trip Cost'] / avg_flight_time, 0)
    results['Annual Cost'] = results['Annual Membership Fees'] + (results['Effective Hourly Rate'] * annual_usage)
    
    # Simulate Lifetime Cost with CPI
    lifetime_cost = np.zeros(len(results))
    for year in range(int(depreciation_term_years)):
        hr_inflated = results['Hourly Rate'] * ((1 + cpi_rate) ** year)
        trip_cost_yr = ((avg_flight_time * hr_inflated) + results['Day Rate'] + (avg_flight_time * results['Amoritized management fee']))
        trip_cost_fet_yr = np.where(results['Program Type'] == 'JC', trip_cost_yr * (1 + tax_rate), trip_cost_yr)
        total_trip_cost_yr = trip_cost_fet_yr + (results['Depreciation Hourly'] * avg_flight_time)
        ehr_yr = np.where(avg_flight_time > 0, total_trip_cost_yr / avg_flight_time, 0)
        lifetime_cost += results['Annual Membership Fees'] + (ehr_yr * annual_usage)
        
    results['Total Lifetime Cost'] = lifetime_cost
    return results

def calculate_costs_jc(df, avg_flight_time, annual_usage_jc, tax_rate, depreciation_term_years, cpi_rate):
    results = df.copy()
    results['Billed Hours'] = results['Mins'].apply(lambda x: max(x, avg_flight_time))
    
    # Base Year Calculations
    results['Total Hourly Rate'] = results['Hourly'] + results['Fuel Surcharges']
    results['Base Trip Cost'] = (results['Billed Hours'] * results['Total Hourly Rate']) + results['Daily']
    results['Trip Cost (FET)'] = results['Base Trip Cost'] * (1 + tax_rate)
    results['Hourly Membership Cost'] = results['Annual Membership Fees'] / annual_usage_jc
    results['Total Trip Cost'] = results['Trip Cost (FET)'] + (results['Hourly Membership Cost'] * avg_flight_time)
    results['Effective Hourly Rate'] = np.where(avg_flight_time > 0, results['Total Trip Cost'] / avg_flight_time, 0)
    results['Annual Cost'] = results['Effective Hourly Rate'] * annual_usage_jc
    
    # Simulate Lifetime Cost with CPI 
    lifetime_cost = np.zeros(len(results))
    for year in range(int(depreciation_term_years)):
        hr_inflated = results['Hourly'] * ((1 + cpi_rate) ** year)
        total_hr_yr = hr_inflated + results['Fuel Surcharges']
        base_trip_cost_yr = (results['Billed Hours'] * total_hr_yr) + results['Daily']
        trip_cost_fet_yr = base_trip_cost_yr * (1 + tax_rate)
        total_trip_cost_yr = trip_cost_fet_yr + (results['Hourly Membership Cost'] * avg_flight_time)
        ehr_yr = np.where(avg_flight_time > 0, total_trip_cost_yr / avg_flight_time, 0)
        lifetime_cost += ehr_yr * annual_usage_jc

    results['Total Lifetime Cost'] = lifetime_cost
    return results

# Highlight Function for Dataframes
def highlight_flyexclusive(row):
    color = 'background-color: rgba(255, 215, 0, 0.4)' # Gold/Yellow highlight
    if 'flyexclusive' in str(row['Company']).lower():
        return [color] * len(row)
    return [''] * len(row)

# --- 4. DATA INITIALIZATION & SESSION STATE ---
df_frac = load_data_fractional()
df_jc = load_data_jc()
if df_frac.empty or df_jc.empty:
    st.warning("Please ensure both 'raw data.csv' and 'jc raw data.csv' are in the directory.")
    st.stop()

all_cabins = sorted(list(set(df_frac['Cabin type'].unique()) | set(df_jc['Cabin type'].unique())))
cabin_options = ["All"] + all_cabins

# Initialize Session State for Forecasting Custom Rows (Empty by default)
if 'custom_forecast' not in st.session_state:
    st.session_state['custom_forecast'] = pd.DataFrame(columns=[
        'Program Name', 'Cabin type', 'Hourly', 'Daily', 'Mins', 'Fuel Surcharges', 'Annual Membership Fees'
    ])

# --- 5. DYNAMIC SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("Navigation")
    analysis_mode = st.radio("Select Analysis Mode", [
        "Fractional Ownership", 
        "Jet Club / Card Programs", 
        "Compare: Fractional vs Jet Club"
    ])
    st.markdown("---")
    
    st.header("1. Shared Inputs")
    avg_flight_time = st.number_input("Average Flight Time (Hours)", min_value=0.5, value=2.0, step=0.1)
    selected_cabin = st.selectbox("Preferred Cabin Type", cabin_options, index=0)
    st.markdown("---")

    # Conditional Inputs based on Analysis Mode
    if analysis_mode == "Fractional Ownership":
        st.header("2. Fractional Inputs")
        available_hours = sorted([int(x) for x in df_frac['Fractional Hours'].unique() if x > 0])
        if not available_hours: available_hours = [50, 75, 100, 150, 200]
        frac_annual_usage = st.selectbox("Annual Usage (Fractional Hours Tier)", available_hours, index=0)
        depreciation_pct_new = st.slider("Depreciation - New Aircraft (%)", 0, 100, 20)
        depreciation_pct_used = st.slider("Depreciation - Secondary Aircraft (%)", 0, 100, 30)
    
    elif analysis_mode == "Jet Club / Card Programs":
        st.header("2. Jet Club Inputs")
        jc_annual_usage = st.slider("Annual Estimated Usage (Hours)", min_value=10, max_value=500, value=50, step=10)

    elif analysis_mode == "Compare: Fractional vs Jet Club":
        st.header("2. Comparison Inputs")
        st.markdown("Select an assumed usage tier to compare apples-to-apples.")
        available_hours = sorted([int(x) for x in df_frac['Fractional Hours'].unique() if x > 0])
        if not available_hours: available_hours = [50, 75, 100, 150, 200]
        compare_annual_usage = st.selectbox("Assumed Annual Usage (Hours)", available_hours, index=0)
        depreciation_pct_new = st.slider("Depreciation - New Aircraft (%)", 0, 100, 20)
        depreciation_pct_used = st.slider("Depreciation - Used Aircraft (%)", 0, 100, 30)

    # UI Update: Form-based Forecasting Tools
    if analysis_mode in ["Jet Club / Card Programs", "Compare: Fractional vs Jet Club"]:
        st.markdown("---")
        st.header("🚀 Custom Forecast Scenarios")
        st.markdown("Easily add hypothetical **flyExclusive** pricing scenarios below.")
        
        with st.form("forecast_form"):
            st.subheader("Create New Tier")
            f_name = st.text_input("Program Name", value="Forecast JC Tier")
            f_cabin = st.selectbox("Cabin Type", options=all_cabins[1:]) # Skip "All"
            
            col1, col2 = st.columns(2)
            f_hourly = col1.number_input("Hourly Rate ($)", min_value=0.0, value=4000.0, step=100.0)
            f_daily = col2.number_input("Daily Rate ($)", min_value=0.0, value=7000.0, step=100.0)
            
            col3, col4 = st.columns(2)
            f_mins = col3.number_input("Minimum Hours", min_value=0.0, value=1.2, step=0.1)
            f_fuel = col4.number_input("Fuel Surcharge ($)", min_value=0.0, value=0.0, step=50.0)
            
            f_mem = st.number_input("Annual Membership Fee ($)", min_value=0.0, value=15000.0, step=500.0)
            
            submit_forecast = st.form_submit_button("➕ Add Scenario")
            
            if submit_forecast:
                new_row = pd.DataFrame({
                    'Program Name': [f_name],
                    'Cabin type': [f_cabin],
                    'Hourly': [f_hourly],
                    'Daily': [f_daily],
                    'Mins': [f_mins],
                    'Fuel Surcharges': [f_fuel],
                    'Annual Membership Fees': [f_mem]
                })
                st.session_state['custom_forecast'] = pd.concat([st.session_state['custom_forecast'], new_row], ignore_index=True)
                st.success(f"Added {f_name} to the analysis!")
        
        # Display current forecasts and allow deletion
        if not st.session_state['custom_forecast'].empty:
            st.markdown("**Active Forecasts (Edit/Delete)**")
            st.session_state['custom_forecast'] = st.data_editor(
                st.session_state['custom_forecast'],
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True
            )
            if st.button("🗑️ Clear All Forecasts"):
                st.session_state['custom_forecast'] = pd.DataFrame(columns=[
                    'Program Name', 'Cabin type', 'Hourly', 'Daily', 'Mins', 'Fuel Surcharges', 'Annual Membership Fees'
                ])
                st.rerun()

    st.markdown("---")
    st.header("3. System Definitions")
    with st.expander("Edit Logic Parameters"):
        tax_rate = st.number_input("FET Tax Rate (Decimal)", value=0.075, format="%.3f")
        dep_term = st.number_input("Depreciation Term (Years)", value=5, step=1)
        cpi_rate = st.number_input("Annual CPI (Hourly Rate Increase)", value=0.03, format="%.3f")

# --- 6. DATA MERGING (Inject Forecasts) ---
df_jc_combined = df_jc.copy()

if analysis_mode in ["Jet Club / Card Programs", "Compare: Fractional vs Jet Club"]:
    custom_df = st.session_state['custom_forecast'].copy()
    if not custom_df.empty:
        # Clean numeric columns before merging
        num_cols_custom = ['Mins', 'Daily', 'Hourly', 'Fuel Surcharges', 'Annual Membership Fees']
        for col in num_cols_custom:
            custom_df[col] = pd.to_numeric(custom_df[col], errors='coerce').fillna(0)
        
        # Assign Company so it gets highlighted automatically
        custom_df['Company'] = 'flyExclusive (Forecast)' 
        df_jc_combined = pd.concat([df_jc_combined, custom_df], ignore_index=True)


# --- 7. MAIN PAGE RENDERING ---
st.title("flyExclusive Price Analyzer")

# ==========================================
# MODE 1: FRACTIONAL OWNERSHIP
# ==========================================
if analysis_mode == "Fractional Ownership":
    st.header("📊 Fractional Ownership Analysis")
    
    filtered_frac = df_frac.copy()
    if selected_cabin != "All": filtered_frac = filtered_frac[filtered_frac['Cabin type'] == selected_cabin]
    filtered_frac = filtered_frac[(filtered_frac['Fractional Hours'] == 0) | (filtered_frac['Fractional Hours'] == frac_annual_usage)]

    res_frac = calculate_costs_fractional(filtered_frac, avg_flight_time, frac_annual_usage, depreciation_pct_new, depreciation_pct_used, tax_rate, dep_term, cpi_rate)

    st.subheader("Single Scenario Estimates (Year 1 Basis)")
    display_cols_frac = ['Company', 'type', 'Cabin type', 'Program Type', 'Fractional Hours', 'Share %', 'Purchase Price', 
                         'Hourly Rate', 'Depreciation Hourly', 'Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost']
    
    styled_frac_df = res_frac[display_cols_frac].style.apply(highlight_flyexclusive, axis=1).format({
        'Share %': '{:.2%}', 'Purchase Price': '${:,.0f}', 'Fractional Hours': '{:.0f}',
        'Hourly Rate': '${:,.0f}', 'Depreciation Hourly': '${:,.0f}',
        'Effective Hourly Rate': '${:,.0f}', 'Annual Cost': '${:,.0f}', 'Total Lifetime Cost': '${:,.0f}'
    })
    st.dataframe(styled_frac_df, use_container_width=True)

    if not res_frac.empty:
        st.subheader("Average Annual Cost by Company")
        avg_annual_frac = res_frac.groupby('Company')['Annual Cost'].mean().reset_index()
        avg_annual_frac['highlight'] = avg_annual_frac['Company'].apply(lambda x: 'flyExclusive' if 'flyexclusive' in str(x).lower() else 'Other')
        fig_bar_frac = px.bar(avg_annual_frac, x='Company', y='Annual Cost', color='highlight',
                              color_discrete_map={'flyExclusive': '#ff4b4b', 'Other': '#1f77b4'},
                              labels={'Annual Cost': 'Average Annual Cost ($)'})
        fig_bar_frac.update_layout(showlegend=False)
        st.plotly_chart(fig_bar_frac, use_container_width=True)

    st.subheader("🔥 Effective Hourly Rate Heatmap")
    if not filtered_frac.empty:
        flight_times = np.arange(1.0, 5.5, 0.5) 
        hm_data_frac = []
        for ft in flight_times:
            temp_df = calculate_costs_fractional(filtered_frac, ft, frac_annual_usage, depreciation_pct_new, depreciation_pct_used, tax_rate, dep_term, cpi_rate)
            for _, row in temp_df.iterrows():
                cabin = str(row['Cabin type']).lower()
                if cabin == 'light' and ft >= 3.5: continue
                if cabin == 'mid' and ft >= 4.5: continue
                hm_data_frac.append({'Company': row['Company'], 'Cabin type': row['Cabin type'], 'Flight Time': ft, 'Effective Hourly Rate': row['Effective Hourly Rate']})

        if hm_data_frac:
            hm_df_frac = pd.DataFrame(hm_data_frac)
            pivot_frac = hm_df_frac.pivot_table(index=['Cabin type', 'Company'], columns='Flight Time', values='Effective Hourly Rate', aggfunc='mean')
            pivot_frac.sort_index(level=['Cabin type', 'Company'], inplace=True)
            pivot_frac.index = [f"⭐ {cab} | {comp}" if 'flyexclusive' in comp.lower() else f"{cab} | {comp}" for cab, comp in pivot_frac.index]
            
            # --- Dynamic Height Calculation (Fractional) ---
            n_rows = len(pivot_frac)
            dynamic_height = max(n_rows * 35 + 120, 400) # 35px per row + buffer
            
            fig_hm_frac = px.imshow(pivot_frac, text_auto=".0f", aspect="auto", color_continuous_scale="Blues")
            fig_hm_frac.update_xaxes(side="top")
            fig_hm_frac.update_layout(height=dynamic_height)
            st.plotly_chart(fig_hm_frac, use_container_width=True)


# ==========================================
# MODE 2: JET CLUB
# ==========================================
elif analysis_mode == "Jet Club / Card Programs":
    st.header("💳 Jet Club / Card Programs Analysis")
    
    filtered_jc = df_jc_combined.copy() # Use the combined df containing forecasts
    if selected_cabin != "All": filtered_jc = filtered_jc[filtered_jc['Cabin type'] == selected_cabin]

    res_jc = calculate_costs_jc(filtered_jc, avg_flight_time, jc_annual_usage, tax_rate, dep_term, cpi_rate)

    st.subheader("Single Scenario Estimates (Year 1 Basis)")
    display_cols_jc = ['Program Name', 'Company', 'Cabin type', 'Mins', 'Daily', 'Hourly', 
                       'Fuel Surcharges', 'Annual Membership Fees', 'Effective Hourly Rate', 'Annual Cost']
    
    styled_jc_df = res_jc[display_cols_jc].sort_values(by='Effective Hourly Rate').style.apply(highlight_flyexclusive, axis=1).format({
        'Mins': '{:.1f}', 'Daily': '${:,.0f}', 'Hourly': '${:,.0f}', 
        'Fuel Surcharges': '${:,.0f}', 'Annual Membership Fees': '${:,.0f}',
        'Effective Hourly Rate': '${:,.0f}', 'Annual Cost': '${:,.0f}'
    })
    st.dataframe(styled_jc_df, use_container_width=True)

    if not res_jc.empty:
        st.subheader("Average Effective Hourly Rate by Company (Sorted)")
        avg_hr_jc = res_jc.groupby('Company')['Effective Hourly Rate'].mean().reset_index()
        avg_hr_jc = avg_hr_jc.sort_values('Effective Hourly Rate', ascending=True)
        avg_hr_jc['highlight'] = avg_hr_jc['Company'].apply(lambda x: 'flyExclusive' if 'flyexclusive' in str(x).lower() else 'Other')
        
        fig_bar_jc = px.bar(avg_hr_jc, x='Company', y='Effective Hourly Rate', color='highlight',
                            color_discrete_map={'flyExclusive': '#ff4b4b', 'Other': '#1f77b4'},
                            labels={'Effective Hourly Rate': 'Avg Effective Hourly Rate ($)'})
        fig_bar_jc.update_layout(showlegend=False)
        st.plotly_chart(fig_bar_jc, use_container_width=True)

    st.subheader("🔥 Effective Hourly Rate Heatmap")
    if not filtered_jc.empty:
        flight_times = np.arange(1.0, 5.5, 0.5) 
        hm_data_jc = []
        for ft in flight_times:
            temp_df = calculate_costs_jc(filtered_jc, ft, jc_annual_usage, tax_rate, dep_term, cpi_rate)
            for _, row in temp_df.iterrows():
                cabin = str(row['Cabin type']).lower()
                if cabin == 'light' and ft >= 3.5: continue
                if cabin == 'mid' and ft >= 4.5: continue
                prog_name = row['Program Name'] if pd.notna(row['Program Name']) else row['Company']
                hm_data_jc.append({'Program': prog_name, 'Cabin type': row['Cabin type'], 'Company': row['Company'],
                                   'Flight Time': ft, 'Effective Hourly Rate': row['Effective Hourly Rate']})

        if hm_data_jc:
            hm_df_jc = pd.DataFrame(hm_data_jc)
            pivot_jc = hm_df_jc.pivot_table(index=['Cabin type', 'Program', 'Company'], columns='Flight Time', values='Effective Hourly Rate', aggfunc='mean')
            pivot_jc = pivot_jc.reset_index()
            pivot_jc['Label'] = pivot_jc.apply(lambda x: f"⭐ {x['Cabin type']} | {x['Program']}" if 'flyexclusive' in str(x['Company']).lower() else f"{x['Cabin type']} | {x['Program']}", axis=1)
            pivot_jc.set_index('Label', inplace=True)
            pivot_jc.sort_index(inplace=True)
            pivot_jc.drop(columns=['Cabin type', 'Program', 'Company'], inplace=True, errors='ignore')
            
            # --- Dynamic Height Calculation (Jet Club) ---
            n_rows = len(pivot_jc)
            dynamic_height = max(n_rows * 35 + 120, 400) # 35px per row + buffer
            
            fig_hm_jc = px.imshow(pivot_jc, text_auto=".0f", aspect="auto", color_continuous_scale="Blues")
            fig_hm_jc.update_xaxes(side="top")
            fig_hm_jc.update_layout(height=dynamic_height)
            st.plotly_chart(fig_hm_jc, use_container_width=True)


# ==========================================
# MODE 3: COMPARISON (FRACTIONAL VS JC)
# ==========================================
elif analysis_mode == "Compare: Fractional vs Jet Club":
    st.header("⚖️ flyExclusive: Fractional vs Jet Club")
    st.markdown(f"**Assumed Scenario:** {compare_annual_usage} Flight Hours/Year | {avg_flight_time} Avg Flight Time")
    
    fe_frac = df_frac[df_frac['Company'].str.contains('flyexclusive', case=False, na=False)].copy()
    
    # Use the combined df containing forecasts
    fe_jc = df_jc_combined[df_jc_combined['Company'].str.contains('flyexclusive', case=False, na=False)].copy()

    if selected_cabin != "All":
        fe_frac = fe_frac[fe_frac['Cabin type'] == selected_cabin]
        fe_jc = fe_jc[fe_jc['Cabin type'] == selected_cabin]
        
    fe_frac = fe_frac[(fe_frac['Fractional Hours'] == 0) | (fe_frac['Fractional Hours'] == compare_annual_usage)]

    res_frac = calculate_costs_fractional(fe_frac, avg_flight_time, compare_annual_usage, depreciation_pct_new, depreciation_pct_used, tax_rate, dep_term, cpi_rate)
    res_jc = calculate_costs_jc(fe_jc, avg_flight_time, compare_annual_usage, tax_rate, dep_term, cpi_rate)

    if not res_frac.empty:
        res_frac['Model'] = 'Fractional'
        res_frac['Display Name'] = res_frac['Cabin type'] + " " + res_frac['type'].str.capitalize() + " (" + res_frac['Program Type'] + ")"
    if not res_jc.empty:
        res_jc['Model'] = np.where(res_jc['Company'].str.contains('Forecast'), 'Jet Club (Forecast)', 'Jet Club')
        res_jc['Display Name'] = res_jc['Program Name']

    compare_cols = ['Display Name', 'Model', 'Cabin type', 'Effective Hourly Rate', 'Annual Cost', 'Total Lifetime Cost']
    
    combo_df = pd.DataFrame()
    if not res_frac.empty and not res_jc.empty:
        combo_df = pd.concat([res_frac[compare_cols], res_jc[compare_cols]], ignore_index=True)
    elif not res_frac.empty:
        combo_df = res_frac[compare_cols]
    elif not res_jc.empty:
        combo_df = res_jc[compare_cols]

    if combo_df.empty:
        st.warning("No flyExclusive data available for the selected filters.")
    else:
        st.subheader("Side-by-Side Financials (Year 1 Basis + Lifetime Compounding)")
        combo_df['Company'] = 'flyExclusive' 
        
        styled_combo = combo_df.sort_values(by='Total Lifetime Cost').style.apply(highlight_flyexclusive, axis=1).format({
            'Effective Hourly Rate': '${:,.0f}', 'Annual Cost': '${:,.0f}', 'Total Lifetime Cost': '${:,.0f}'
        })
        st.dataframe(styled_combo, use_container_width=True)

        st.subheader(f"{dep_term}-Year Total Lifetime Cost Comparison (Includes CPI)")
        fig_comp_bar = px.bar(combo_df, x='Display Name', y='Total Lifetime Cost', color='Model',
                              color_discrete_map={'Fractional': '#1f77b4', 'Jet Club': '#ff4b4b', 'Jet Club (Forecast)': '#ff9896'},
                              text_auto='$.2s')
        fig_comp_bar.update_layout(xaxis_title="", yaxis_title="Total Lifetime Cost ($)")
        st.plotly_chart(fig_comp_bar, use_container_width=True)

        st.subheader("Flight Time Sensitivity (Effective Hourly Rate)")
        st.markdown("Shows how the Year 1 Effective Hourly Rate shifts based on your specific trip duration.")
        
        flight_times = np.arange(1.0, 5.5, 0.5) 
        sensitivity_data = []

        for ft in flight_times:
            temp_frac = calculate_costs_fractional(fe_frac, ft, compare_annual_usage, depreciation_pct_new, depreciation_pct_used, tax_rate, dep_term, cpi_rate)
            temp_jc = calculate_costs_jc(fe_jc, ft, compare_annual_usage, tax_rate, dep_term, cpi_rate)
            
            for _, row in temp_frac.iterrows():
                name = str(row['Cabin type']) + " " + str(row['type']).capitalize() + " (" + str(row['Program Type']) + ")"
                sensitivity_data.append({'Flight Time': ft, 'Effective Hourly Rate': row['Effective Hourly Rate'], 'Program': name, 'Model': 'Fractional'})
                
            for _, row in temp_jc.iterrows():
                name = row['Program Name'] if pd.notna(row['Program Name']) else "Jet Club Option"
                model_type = 'Jet Club (Forecast)' if 'Forecast' in str(row['Company']) else 'Jet Club'
                sensitivity_data.append({'Flight Time': ft, 'Effective Hourly Rate': row['Effective Hourly Rate'], 'Program': name, 'Model': model_type})
                
        if sensitivity_data:
            sens_df = pd.DataFrame(sensitivity_data)
            fig_sens = px.line(sens_df, x='Flight Time', y='Effective Hourly Rate', color='Program', symbol='Model',
                               markers=True, labels={'Effective Hourly Rate': 'Rate ($)', 'Flight Time': 'Trip Duration (Hours)'})
            fig_sens.update_layout(hovermode="x unified")

            st.plotly_chart(fig_sens, use_container_width=True)
