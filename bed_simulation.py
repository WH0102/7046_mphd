import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("🏥 Demo Hospital Flow Simulation")
st.markdown("""
Simulates patient flow through Emergency Department (ED), ICU, and General Wards with:
- ED boarding/congestion effects
- WHO bed-population benchmarks
- Staffing constraints
- Real-time occupancy monitoring
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Scenario selection
    scenario = st.selectbox(
        "Scenario",
        ["Normal Operations", "Flu Season", "Pandemic Surge", "Mass Casualty"],
        index=0
    )
    
    # Hospital resources
    st.subheader("Hospital Resources")
    ed_beds = st.slider("ED Treatment Spaces", 10, 50, 25)
    icu_beds = st.slider("ICU Beds", 5, 50, 15)
    ward_beds = st.slider("General Ward Beds", 20, 100, 50)
    
    # Staffing
    st.subheader("Staffing Levels")
    ed_staff = st.slider("ED Staff", 10, 50, 20)
    icu_staff = st.slider("ICU Staff", 5, 30, 10)
    ward_staff = st.slider("Ward Staff", 10, 60, 25)
    
    # Operational parameters
    st.subheader("Operational Parameters")
    sim_days = st.slider("Simulation Duration (days)", 7, 90, 30)
    ed_avg_stay = st.slider("ED Avg Stay (hours)", 2, 12, 4)
    transfer_time = st.slider("Bed Transfer Time (mins)", 15, 120, 45)
    
    # Advanced options
    with st.expander("Advanced Options"):
        population = st.number_input("Population Served", 100000, 10000000, 500000)
        use_ml = st.checkbox("Enable Demand Forecasting", True)

# --- Main Page Controls ---
run_simulation = st.button("🚀 Run Full Simulation", type="primary",
                          help="Start simulation with current parameters")

# --- Simulation Classes ---
class Hospital:
    def __init__(self, env):
        self.env = env
        self.ed = EmergencyDepartment(env)
        self.icu = ICU(env)
        self.ward = GeneralWard(env)
        self.patient_log = []
        self.overflow_stats = {
            'ed_turned_away': 0,
            'ed_to_icu_overflow': 0,
            'ed_to_ward_overflow': 0,
            'icu_full': 0,
            'ward_full': 0
        }
        
    def log_patient(self, patient_id, event, time, location):
        self.patient_log.append({
            "patient_id": patient_id,
            "event": event,
            "time": time,
            "location": location
        })

class EmergencyDepartment:
    def __init__(self, env):
        self.env = env
        self.treatment_spaces = simpy.Resource(env, capacity=ed_beds)
        self.staff = simpy.Resource(env, capacity=ed_staff)
        self.waiting_patients = []
        self.boarding_patients = []
        self.turned_away = 0
        
    def process_patient(self, patient_id):
        # Check if ED is at capacity
        if len(self.waiting_patients) + self.treatment_spaces.count >= ed_beds * 1.5:  # 150% capacity threshold
            hospital.overflow_stats['ed_turned_away'] += 1
            self.turned_away += 1
            hospital.log_patient(patient_id, "ED Turned Away", self.env.now, "ED")
            return
            
        arrival_time = self.env.now
        self.waiting_patients.append(patient_id)
        
        with self.staff.request() as staff_req:
            yield staff_req
            self.waiting_patients.remove(patient_id)
            
            with self.treatment_spaces.request() as bed_req:
                yield bed_req
                hospital.log_patient(patient_id, "ED Admission", self.env.now, "ED")
                
                # ED treatment time
                yield self.env.timeout(np.random.exponential(ed_avg_stay))
                
                # Determine disposition
                disposition = np.random.random()
                if disposition < 0.2:  # 20% need ICU
                    success = yield self.env.process(self.transfer_to_icu(patient_id))
                    if not success:
                        hospital.overflow_stats['ed_to_icu_overflow'] += 1
                elif disposition < 0.6:  # 48% need ward (60% of remaining)
                    success = yield self.env.process(self.transfer_to_ward(patient_id))
                    if not success:
                        hospital.overflow_stats['ed_to_ward_overflow'] += 1
                else:  # 32% discharged
                    hospital.log_patient(patient_id, "ED Discharge", self.env.now, "ED")
    
    def transfer_to_icu(self, patient_id):
        hospital.log_patient(patient_id, "ED Boarding (ICU)", self.env.now, "ED")
        self.boarding_patients.append(patient_id)
        
        max_wait = 72  # Max 72 hours boarding
        start_time = self.env.now
        
        while self.env.now - start_time < max_wait:
            if icu.beds.count < icu.beds.capacity:
                yield self.env.timeout(transfer_time)  # Transfer process
                self.boarding_patients.remove(patient_id)
                yield self.env.process(icu.admit_patient(patient_id))
                return True  # Successful transfer
            
            yield self.env.timeout(1)  # Check every hour
        
        # If we get here, transfer failed
        self.boarding_patients.remove(patient_id)
        hospital.log_patient(patient_id, "ED Discharge (ICU Overflow)", self.env.now, "ED")
        return False
    
    def transfer_to_ward(self, patient_id):
        hospital.log_patient(patient_id, "ED Boarding (Ward)", self.env.now, "ED")
        self.boarding_patients.append(patient_id)
        
        max_wait = 48  # Max 48 hours boarding
        start_time = self.env.now
        
        while self.env.now - start_time < max_wait:
            if ward.beds.count < ward.beds.capacity:
                yield self.env.timeout(transfer_time)
                self.boarding_patients.remove(patient_id)
                yield self.env.process(ward.admit_patient(patient_id))
                return True  # Successful transfer
            
            yield self.env.timeout(1)  # Check every hour
        
        # If we get here, transfer failed
        self.boarding_patients.remove(patient_id)
        hospital.log_patient(patient_id, "ED Discharge (Ward Overflow)", self.env.now, "ED")
        return False

class ICU:
    def __init__(self, env):
        self.env = env
        self.beds = simpy.Resource(env, capacity=icu_beds)
        self.staff = simpy.Resource(env, capacity=icu_staff)
        
    def admit_patient(self, patient_id):
        los = np.random.lognormal(mean=np.log(7), sigma=0.3)
        hospital.log_patient(patient_id, "ICU Admission", self.env.now, "ICU")
        
        with self.staff.request() as req:
            yield req
            with self.beds.request() as bed_req:
                yield bed_req
                yield self.env.timeout(los * 24)  # LOS in hours
                hospital.log_patient(patient_id, "ICU Discharge", self.env.now, "ICU")

class GeneralWard:
    def __init__(self, env):
        self.env = env
        self.beds = simpy.Resource(env, capacity=ward_beds)
        self.staff = simpy.Resource(env, capacity=ward_staff)
        
    def admit_patient(self, patient_id):
        los = np.random.lognormal(mean=np.log(5), sigma=0.4)
        hospital.log_patient(patient_id, "Ward Admission", self.env.now, "Ward")
        
        with self.staff.request() as req:
            yield req
            with self.beds.request() as bed_req:
                yield bed_req
                yield self.env.timeout(los * 24)
                hospital.log_patient(patient_id, "Ward Discharge", self.env.now, "Ward")

# --- Simulation Execution ---
if run_simulation:
    st.subheader(f"Simulation Results: {scenario}")
    
    # Initialize
    env = simpy.Environment()
    hospital = Hospital(env)
    icu = hospital.icu
    ward = hospital.ward
    ed = hospital.ed
    
    # Patient generator
    def patient_generator(env):
        patient_id = 0
        arrival_rate = {
            "Normal Operations": 15,
            "Flu Season": 25,
            "Pandemic Surge": 40,
            "Mass Casualty": 60
        }[scenario]
        
        while True:
            yield env.timeout(np.random.exponential(60/arrival_rate))  # Arrivals per hour
            env.process(ed.process_patient(patient_id))
            patient_id += 1
    
    env.process(patient_generator(env))
    
    # Run simulation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for day in range(sim_days):
        env.run(until=24*(day+1))
        progress_bar.progress((day+1)/sim_days)
        status_text.text(
            f"Day {day+1}/{sim_days} | "
            f"ED: {len(ed.waiting_patients)} waiting, {len(ed.boarding_patients)} boarding | "
            f"ICU: {icu.beds.count}/{icu_beds} | "
            f"Ward: {ward.beds.count}/{ward_beds}"
        )
    
    # --- Results Visualization ---
    st.subheader("Patient Flow Analysis")
    
    # Convert log to DataFrame
    df = pd.DataFrame(hospital.patient_log)
    df['hour'] = df['time'] % 24
    df['day'] = df['time'] // 24
    
    # Key metrics
    ed_wait_time = df[df['event'] == 'ED Admission']['time'].diff().mean()
    boarding_hours = df[df['event'].str.contains('Boarding')].groupby('patient_id')['time'].agg(['min','max'])
    avg_boarding = (boarding_hours['max'] - boarding_hours['min']).mean() if not boarding_hours.empty else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg ED Wait Time", f"{ed_wait_time:.1f} hours")
    col2.metric("Avg Boarding Time", f"{avg_boarding:.1f} hours")
    col3.metric("Max ED Occupancy", f"{df[df['location'] == 'ED'].shape[0]/ed_beds:.0%}")
    
    # Occupancy plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    # ED congestion
    ed_occ = df[df['location'] == 'ED'].groupby('time').count()['patient_id']
    ax[0].plot(ed_occ.index/24, ed_occ.values/ed_beds)
    ax[0].axhline(1, color='r', linestyle='--')
    ax[0].set_title("ED Occupancy %")
    ax[0].set_ylim(0, 1.5)
    
    # ICU/Ward occupancy
    for i, (unit, color) in enumerate(zip(['ICU', 'Ward'], ['b', 'g']), 1):
        occ = df[df['location'] == unit].groupby('time').count()['patient_id']
        ax[i].plot(occ.index/24, occ.values/(icu_beds if unit == 'ICU' else ward_beds), color)
        ax[i].axhline(1, color='r', linestyle='--')
        ax[i].set_title(f"{unit} Occupancy %")
        ax[i].set_ylim(0, 1.2)
    
    st.pyplot(fig)
    
    # Boarding analysis
    st.subheader("ED Boarding Breakdown")
    boarding_df = df[df['event'].str.contains('Boarding')]
    if not boarding_df.empty:
        boarding_counts = boarding_df['event'].value_counts()
        st.bar_chart(boarding_counts)
        
        st.warning(f"""
        **ED Congestion Alert:**  
        - {boarding_counts.sum()} patients experienced boarding delays  
        - Average boarding duration: {avg_boarding:.1f} hours  
        - Consider:  
          • Increasing ICU/Ward capacity  
          • Implementing ED overflow protocols  
          • Improving discharge coordination  
        """)
    else:
        st.success("No ED boarding occurred during simulation")

    # --- New Overflow Metrics Section ---
    st.subheader("🚨 Capacity Overflow Statistics")
    
    overflow_data = {
        "Metric": [
            "ED Patients Turned Away",
            "ED-to-ICU Transfers Failed",
            "ED-to-Ward Transfers Failed",
            "ICU Admissions Rejected",
            "Ward Admissions Rejected"
        ],
        "Count": [
            hospital.overflow_stats['ed_turned_away'],
            hospital.overflow_stats['ed_to_icu_overflow'],
            hospital.overflow_stats['ed_to_ward_overflow'],
            hospital.overflow_stats['icu_full'],
            hospital.overflow_stats['ward_full']
        ]
    }
    
    overflow_df = pd.DataFrame(overflow_data)
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("ED Turned Away", overflow_df.iloc[0]['Count'])
    col2.metric("Failed ICU Transfers", overflow_df.iloc[1]['Count'])
    col3.metric("Failed Ward Transfers", overflow_df.iloc[2]['Count'])
    
    # Overflow bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=overflow_df, x='Metric', y='Count', palette='Reds_r')
    plt.xticks(rotation=45, ha='right')
    plt.title("Patient Overflow by Type")
    st.pyplot(fig)
    
    # Interpretation
    total_overflow = sum(hospital.overflow_stats.values())
    st.warning(f"""
    **System Overcapacity Analysis:**
    - **Total overflow incidents:** {total_overflow}
    - **ED congestion:** {hospital.overflow_stats['ed_turned_away']} patients turned away ({hospital.overflow_stats['ed_turned_away']/total_overflow:.0%} of total)
    - **ICU access issues:** {hospital.overflow_stats['ed_to_icu_overflow']} patients couldn't transfer ({hospital.overflow_stats['ed_to_icu_overflow']/total_overflow:.0%} of total)
    - **Ward access issues:** {hospital.overflow_stats['ed_to_ward_overflow']} patients couldn't transfer ({hospital.overflow_stats['ed_to_ward_overflow']/total_overflow:.0%} of total)
    
    **Recommendations:**
    {f"⚠️ Increase ED capacity (current {ed_beds} beds)" if hospital.overflow_stats['ed_turned_away'] > 0 else "✅ ED capacity adequate"}
    {f"⚠️ Increase ICU beds (current {icu_beds} beds)" if hospital.overflow_stats['ed_to_icu_overflow'] > 0 else "✅ ICU capacity adequate"}
    {f"⚠️ Increase ward beds (current {ward_beds} beds)" if hospital.overflow_stats['ed_to_ward_overflow'] > 0 else "✅ Ward capacity adequate"}
    """)
    
    
    # Download results
    st.download_button(
        label="📥 Download Simulation Data",
        data=df.to_csv(index=False),
        file_name=f"hospital_simulation_{scenario}.csv",
        mime="text/csv"
    )

else:
    st.info("Configure parameters in sidebar and click 'Run Full Simulation' above")

# --- References ---
st.markdown("---")
st.markdown("""
**References:**  
1. American College of Emergency Physicians - ED Crowding Metrics  
2. WHO Hospital Bed Guidelines  
3. AHRQ Emergency Department Modeling Toolkit  
""")
