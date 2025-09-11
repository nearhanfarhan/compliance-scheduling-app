import streamlit as st
import pandas as pd
from datetime import datetime
import gspread
from gspread_dataframe import get_as_dataframe
from geopy.distance import geodesic
import numpy as np
from scipy.optimize import linear_sum_assignment
import requests
from google.oauth2.service_account import Credentials
import io

# =======================================
# App Configuration & UI Setup
# =======================================

st.set_page_config(page_title="Weekly Team Scheduler", layout="wide")
st.title("🚴 Team Scheduling Automation Tool")

# Use Streamlit's secrets management for credentials and API keys
try:
    GOOGLE_MAPS_API_KEY = st.secrets["api_keys"]["google_maps"]
    # Directly use the dictionary from st.secrets for credentials
    SERVICE_ACCOUNT_CREDS = st.secrets["google_credentials"]
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
except (KeyError, FileNotFoundError):
    st.error("🚨 Critical Error: Secrets for Google API or credentials not found. Please configure your `secrets.toml` file.")
    st.stop()


# =======================================
# REFACTORED SCRIPT FUNCTIONS
# (Original logic, adapted for Streamlit)
# =======================================

# --- Caching to prevent re-running expensive operations ---
@st.cache_data(ttl=3600)
def get_gspread_client():
    """Authenticates and returns a gspread client. Cached for performance."""
    creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_CREDS, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=3600)
def fetch_sheet_data(_client, sheet_id, worksheet_name):
    """Fetches and returns data from a Google Sheet worksheet as a DataFrame."""
    try:
        spreadsheet = _client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        return pd.DataFrame(worksheet.get_all_records())
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet with ID '{sheet_id}' not found.")
        return None
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in the spreadsheet.")
        return None

# PART 1 — Generate Weekly Availability
def generate_weekly_availability(daily_roster_files, employee_info_file, client, cyclist_sheet_id):
    """
    Processes uploaded roster files and employee info to generate a weekly availability DataFrame.
    """
    # Load and filter employee data from uploaded file
    employee_df = pd.read_csv(employee_info_file)
    employee_df = employee_df[employee_df['Location'] == 'Lime - London OOW Labor']
    employee_df['Name'] = employee_df['First Name'].str.strip() + ' ' + employee_df['Last Name'].str.strip()
    employee_df = employee_df[['Name', 'Email', 'Role Name']]

    full_schedule = pd.DataFrame()

    def categorise_shift(start_time, shift_date):
        try:
            shift_start = datetime.strptime(start_time, "%I:%M%p")
            is_weekend = shift_date.weekday() in [5, 6]
            if is_weekend:
                return "WEEKEND" if shift_start.hour < 15 else "EVENING"
            else:
                return "DAY" if shift_start.hour < 15 else "EVENING"
        except (ValueError, TypeError):
            return "UNKNOWN"

    # Process uploaded roster files in memory
    for file in daily_roster_files:
        # Infer date from filename
        date_str = file.name.replace('daily-roster-', '').replace('.csv', '')
        try:
            shift_date = datetime.strptime(date_str, '%m_%d_%Y')
            formatted_date = shift_date.strftime('%d/%m')
        except ValueError:
            st.warning(f"Skipping file {file.name}: Date could not be parsed from filename.")
            continue

        df = pd.read_csv(file, header=None, names=['Name', 'Assignment', 'Start Time', 'End Time'])
        df['Name'] = df['Name'].str.strip()
        df['Shift Type'] = df['Start Time'].apply(lambda start: categorise_shift(start, shift_date))
        df[formatted_date] = df['Shift Type']
        df = df.groupby('Name')[formatted_date].first().reset_index()

        if full_schedule.empty:
            full_schedule = df
        else:
            full_schedule = pd.merge(full_schedule, df, on='Name', how='outer')

    if full_schedule.empty:
        st.error("No valid roster files were processed. Cannot continue.")
        return None

    full_schedule = full_schedule.fillna('OFF')
    date_columns = [col for col in full_schedule.columns if col != 'Name']
    sorted_dates = sorted(date_columns, key=lambda x: datetime.strptime(x, '%d/%m'))
    full_schedule = full_schedule[['Name'] + sorted_dates]

    # Merge with cyclist list from Google Sheets
    try:
        spreadsheet = client.open_by_key(cyclist_sheet_id)
        worksheet = spreadsheet.worksheet('Cyclist Details')
        full_cyclist_df = get_as_dataframe(worksheet).dropna(how="all")[['Email', 'Post Code', 'Split']]
    except Exception as e:
        st.error(f"Failed to fetch cyclist details from Google Sheets: {e}")
        return None

    weekly_schedule = pd.merge(employee_df, full_schedule, on='Name', how='right')
    availability_data = pd.merge(weekly_schedule, full_cyclist_df, on='Email', how='left')

    # Reorder columns
    cols = availability_data.columns.tolist()
    if 'Email' in cols:
      email_idx = cols.index('Email')
      new_order = cols[:email_idx+1] + ['Post Code', 'Split'] + [col for col in cols[email_idx+1:] if col not in ['Post Code', 'Split']]
      availability_data = availability_data[new_order]

    return availability_data


# PART 2 — Build Schedule from Availability (Helper Functions)
@st.cache_data
def get_coordinates(postcode, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={postcode},UK&key={api_key}"
    try:
        response = requests.get(url, timeout=5).json()
        if response["status"] == "OK":
            loc = response["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except requests.exceptions.RequestException as e:
        st.warning(f"Geocoding request failed for {postcode}: {e}")
    return None

def generate_worker_availability(df, shift_identifier, split_identifier):
    availability = {}
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        email_raw = row.get("Email", "")
        if pd.isna(email_raw) or str(email_raw).strip() == "":
            continue
        email = str(email_raw).strip().lower()

        post_code = row.get("Post Code", "")
        if post_code == "#N/A" or pd.isna(post_code):
            continue

        if row.get("Split") == split_identifier:
            worker_availability = {
                date: True
                for date in df.columns[5:] # Adjusted index for safety
                if str(row.get(date, "")).startswith(shift_identifier)
            }
            if worker_availability:
                coords = get_coordinates(post_code, GOOGLE_MAPS_API_KEY)
                if coords:
                    availability[email] = {
                        "name": row.get("Name", ""),
                        "post_code": post_code,
                        "availability": worker_availability,
                        "coordinates": coords,
                        "split": row.get("Split")
                    }
    return availability

def generate_reliable_cyclists(df):
    reliable_cyclists = {}
    day_columns = [col for col in df.columns if col not in ["Email", "First Name", "Last Name", "Split"]]
    for _, row in df.iterrows():
        email = row["Email"].strip().lower()
        name = f"{row['First Name']} {row['Last Name']}"
        daily_assignments = {day: row[day] for day in day_columns if pd.notna(row[day]) and row[day] != ""}
        reliable_cyclists[email] = {"name": name, "daily_assignments": daily_assignments}
    return reliable_cyclists

def create_zone_mappings(data_df):
    priority_mapping = {1: "very_high", 2: "high", 3: "medium", 4: "low"}
    zones = {key: [] for key in priority_mapping.values()}
    for _, row in data_df.iterrows():
        priority_key = priority_mapping.get(row.get("priority"), "low")
        zones[priority_key].append(
            (row.get("Name"), row.get("latitude"), row.get("longitude"))
        )
    return zones

def compute_distance_matrix(workers, zone_list, availability):
    worker_coords = [availability[w]["coordinates"] for w in workers]
    return np.array([[geodesic(worker, (zone[1], zone[2])).km for zone in zone_list] for worker in worker_coords])

def get_weekday_name(date_str):
    current_year = datetime.today().year
    dt = datetime.strptime(f"{date_str}/{current_year}", "%d/%m/%Y")
    return dt.strftime("%A")

def assign_workers_to_zones(df, availability, zones, reliable_cyclists):
    assignments_per_day = {}
    priority_order = ["very_high", "high", "medium", "low"]
    date_columns = [col for col in df.columns if '/' in str(col)]

    for day in date_columns:
        available_workers = [email for email in availability if day in availability[email]["availability"]]
        remaining_workers = set(available_workers)
        assigned_workers_this_day = set()
        assignments_per_day[day] = []

        # 1. Pre-assign reliable cyclists
        weekday_name = get_weekday_name(day)
        for email, details in reliable_cyclists.items():
            if email in remaining_workers:
                assigned_zone_name = details["daily_assignments"].get(weekday_name)
                if assigned_zone_name:
                    all_zones_flat = [z for p_zones in zones.values() for z in p_zones]
                    assigned_zone = next((z for z in all_zones_flat if z[0] == assigned_zone_name), None)
                    if assigned_zone:
                        distance = geodesic(availability[email]["coordinates"], (assigned_zone[1], assigned_zone[2])).km
                        assignments_per_day[day].append((availability[email]["name"], assigned_zone_name, "pre-assigned", distance))
                        assigned_workers_this_day.add(email)
        
        remaining_workers -= assigned_workers_this_day

        # 2. Assign remaining workers using optimization
        for priority in priority_order:
            current_priority_zones = zones.get(priority, [])
            if not remaining_workers or not current_priority_zones:
                continue

            # Filter out zones that are already assigned
            assigned_zone_names = {a[1] for a in assignments_per_day[day]}
            filtered_zones = [z for z in current_priority_zones if z[0] not in assigned_zone_names]
            
            if not filtered_zones:
                continue
            
            # Match available workers to available zones
            current_workers_list = sorted(list(remaining_workers))
            dist_matrix = compute_distance_matrix(current_workers_list, filtered_zones, availability)
            
            worker_indices, zone_indices = linear_sum_assignment(dist_matrix)

            for i in range(len(worker_indices)):
                worker_email = current_workers_list[worker_indices[i]]
                assigned_zone = filtered_zones[zone_indices[i]]
                distance = dist_matrix[worker_indices[i], zone_indices[i]]
                
                assignments_per_day[day].append((availability[worker_email]["name"], assigned_zone[0], priority, distance))
                assigned_workers_this_day.add(worker_email)
                remaining_workers.remove(worker_email)

        # 3. List unassigned workers
        for email in remaining_workers:
            assignments_per_day[day].append((availability[email]["name"], "NOT ASSIGNED", "N/A", 0.0))
            
    return assignments_per_day

def build_final_schedule_df(*assignments_list):
    """Merges all assignment dictionaries into a single DataFrame for display."""
    all_assignments = {}
    for assignment_dict in assignments_list:
        for date, workers in assignment_dict.items():
            if date not in all_assignments:
                all_assignments[date] = []
            all_assignments[date].extend(workers)
            
    if not all_assignments:
        return pd.DataFrame()

    all_dates = sorted(all_assignments.keys(), key=lambda x: datetime.strptime(x, '%d/%m'))
    all_workers = sorted(list(set(worker[0] for date_assignments in all_assignments.values() for worker in date_assignments)))

    schedule_data = []
    for worker_name in all_workers:
        row = {"Worker Name": worker_name}
        for date in all_dates:
            assignment = next((z[1] for z in all_assignments.get(date, []) if z[0] == worker_name), "")
            row[date] = assignment
        schedule_data.append(row)

    return pd.DataFrame(schedule_data)

# --- NEW FUNCTION TO WRITE TO GOOGLE SHEETS ---
def write_schedule_to_google_sheet(client, sheet_id, availability_df, all_assignment_dicts):
    """
    Builds a complete schedule matrix and writes it to the specified Google Sheet.
    """
    try:
        spreadsheet = client.open_by_key(sheet_id)
        sheet_name = "New Schedule"
        # Delete old sheet if it exists, then create a new one
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            spreadsheet.del_worksheet(worksheet)
        except gspread.exceptions.WorksheetNotFound:
            pass # Sheet didn't exist, which is fine
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="30")

        # Get all dates and workers
        all_dates = sorted(
            list(set(date for d in all_assignment_dicts.values() for date in d.keys())),
            key=lambda x: datetime.strptime(x, '%d/%m')
        )
        all_workers = sorted(
            list(set(assign[0] for d in all_assignment_dicts.values() for assigns in d.values() for assign in assigns))
        )
        
        # Create a mapping of worker names to their split for easy lookup
        worker_split_map = pd.Series(availability_df.Split.values, index=availability_df.Name).to_dict()

        # Build schedule matrix
        schedule_matrix = [["Worker Name", "Split"] + all_dates]
        for worker in all_workers:
            split = worker_split_map.get(worker, "N/A")
            row = [worker, split] + [""] * len(all_dates)
            
            for i, date in enumerate(all_dates):
                # Find assignments for this worker on this date across all shift types
                day_zone = next((z[1] for z in all_assignment_dicts['east_day'].get(date, []) if z[0] == worker), "") or \
                           next((z[1] for z in all_assignment_dicts['west_day'].get(date, []) if z[0] == worker), "")
                
                evening_zone = next((z[1] for z in all_assignment_dicts['east_evening'].get(date, []) if z[0] == worker), "") or \
                               next((z[1] for z in all_assignment_dicts['west_evening'].get(date, []) if z[0] == worker), "")

                weekend_zone = next((z[1] for z in all_assignment_dicts['east_weekend'].get(date, []) if z[0] == worker), "") or \
                               next((z[1] for z in all_assignment_dicts['west_weekend'].get(date, []) if z[0] == worker), "")
                
                # Combine zones into a single string if multiple exist
                all_zones = [z for z in [day_zone, evening_zone, weekend_zone] if z]
                cell_value = " / ".join(all_zones)
                
                row[i + 2] = cell_value # +2 to account for Name and Split columns
            
            schedule_matrix.append(row)

        # Update Google Sheet
        worksheet.update("A1", schedule_matrix)
        return True, None # Success
    except Exception as e:
        return False, str(e) # Failure with error message


# =======================================
# STREAMLIT UI AND MAIN EXECUTION
# =======================================

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("Upload files and provide Google Sheet IDs to generate the schedule.")

# Input fields for Google Sheet IDs
fcl_sheet_id = st.sidebar.text_input(
    "Full Cyclist List & Pre-Assignment Sheet ID", 
    "1IH24J2EpdjJKEuvqQrZSfKaElq-k77XJyoSuD5jfUIY"
)
zone_sheet_id = st.sidebar.text_input(
    "Zone Definitions Sheet ID", 
    "1e6BZP5yWlOHLeks0wDInanKX2U_DA2CC0SbGjjeotZM"
)

st.sidebar.header("📄 File Uploads")
employee_info_file = st.sidebar.file_uploader(
    "1. Upload Team Details CSV",
    type="csv"
)
daily_roster_files = st.sidebar.file_uploader(
    "2. Upload Daily Roster CSVs (Select multiple)",
    type="csv",
    accept_multiple_files=True
)

# --- Main App Body ---
if st.button("🚀 Generate Weekly Schedule"):
    if not employee_info_file or not daily_roster_files:
        st.warning("⚠️ Please upload all required files before generating the schedule.")
    else:
        with st.spinner("Processing... This might take a few moments."):
            client = get_gspread_client()

            # --- Part 1: Generate Availability ---
            st.write("### Step 1: Generating Worker Availability...")
            availability_df = generate_weekly_availability(
                daily_roster_files, employee_info_file, client, fcl_sheet_id
            )
            
            if availability_df is None:
                st.error("Failed to generate availability. Please check the uploaded files and Sheet IDs.")
                st.stop()
            
            st.success("✅ Availability data processed successfully.")
            with st.expander("View Generated Availability Data"):
                st.dataframe(availability_df)
                
            # --- Part 2: Fetching Data & Building Schedule ---
            st.write("### Step 2: Fetching Zone Data and Pre-assignments...")
            
            # Fetch external data from Google Sheets
            reliable_cyclists_df = fetch_sheet_data(client, fcl_sheet_id, "Cyclist Pre-Assignment")
            east_day_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_day_zones")
            east_evening_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_evening_zones")
            east_weekend_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_weekend_zones")
            west_day_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_day_zones")
            west_evening_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_evening_zones")
            west_weekend_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_weekend_zones")
            
            all_dfs = [reliable_cyclists_df, east_day_zones_df, east_evening_zones_df, east_weekend_zones_df,
                       west_day_zones_df, west_evening_zones_df, west_weekend_zones_df]
            if any(df is None for df in all_dfs):
                st.error("Could not fetch required data from Google Sheets. Please check Sheet IDs and worksheet names.")
                st.stop()

            reliable_cyclists = generate_reliable_cyclists(reliable_cyclists_df)
            east_daytime_zones = create_zone_mappings(east_day_zones_df)
            east_evening_zones = create_zone_mappings(east_evening_zones_df)
            east_weekend_zones = create_zone_mappings(east_weekend_zones_df)
            west_daytime_zones = create_zone_mappings(west_day_zones_df)
            west_evening_zones = create_zone_mappings(west_evening_zones_df)
            west_weekend_zones = create_zone_mappings(west_weekend_zones_df)
            st.success("✅ Zone data and pre-assignments loaded.")
            
            st.write("### Step 3: Calculating Optimal Assignments...")
            
            east_day_avail = generate_worker_availability(availability_df, "DAY", "East")
            east_eve_avail = generate_worker_availability(availability_df, "EVENING", "East")
            east_wknd_avail = generate_worker_availability(availability_df, "WEEKEND", "East")
            west_day_avail = generate_worker_availability(availability_df, "DAY", "West")
            west_eve_avail = generate_worker_availability(availability_df, "EVENING", "West")
            west_wknd_avail = generate_worker_availability(availability_df, "WEEKEND", "West")
            
            east_day_assign = assign_workers_to_zones(availability_df, east_day_avail, east_daytime_zones, reliable_cyclists)
            east_eve_assign = assign_workers_to_zones(availability_df, east_eve_avail, east_evening_zones, reliable_cyclists)
            east_wknd_assign = assign_workers_to_zones(availability_df, east_wknd_avail, east_weekend_zones, reliable_cyclists)
            west_day_assign = assign_workers_to_zones(availability_df, west_day_avail, west_daytime_zones, reliable_cyclists)
            west_eve_assign = assign_workers_to_zones(availability_df, west_eve_avail, west_evening_zones, reliable_cyclists)
            west_wknd_assign = assign_workers_to_zones(availability_df, west_wknd_avail, west_weekend_zones, reliable_cyclists)
            st.success("✅ Assignments calculated.")

            # --- Part 4: Displaying the Final Schedule ---
            st.write("### Step 4: Displaying Final Schedule...")
            
            all_assigns = [east_day_assign, east_eve_assign, east_wknd_assign, west_day_assign, west_eve_assign, west_wknd_assign]
            final_schedule_df = build_final_schedule_df(*all_assigns)
            
            st.balloons()
            st.header("🎉 Final Weekly Schedule 🎉")
            st.dataframe(final_schedule_df)

            # --- Part 5: Write schedule to Google Sheets ---
            st.write("### Step 5: Writing to Google Sheets...")
            
            # Group all assignment dictionaries for the writer function
            all_assignment_dicts = {
                'east_day': east_day_assign, 'west_day': west_day_assign,
                'east_evening': east_eve_assign, 'west_evening': west_eve_assign,
                'east_weekend': east_wknd_assign, 'west_weekend': west_wknd_assign
            }

            success, error_message = write_schedule_to_google_sheet(client, fcl_sheet_id, availability_df, all_assignment_dicts)

            if success:
                st.success("✅ Schedule successfully written to 'New Schedule' tab in your Google Sheet!")
            else:
                st.error(f"🚨 Failed to write to Google Sheets: {error_message}")
            
            # --- Display detailed assignments per day (CORRECTED) ---
            st.header("📋 Daily Assignment Details")
            all_dates = sorted(list(final_schedule_df.columns[1:]), key=lambda x: datetime.strptime(x, '%d/%m'))

            for date in all_dates:
                with st.expander(f"**📅 Assignments for {date}**"):
                    assignments_on_date = []
                    
                    # Consolidate assignments for the current date with the corrected 4-variable loop
                    for worker, zone, prio, dist in east_day_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'East', 'Shift': 'Day', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    for worker, zone, prio, dist in west_day_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'West', 'Shift': 'Day', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    for worker, zone, prio, dist in east_eve_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'East', 'Shift': 'Evening', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    for worker, zone, prio, dist in west_eve_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'West', 'Shift': 'Evening', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    for worker, zone, prio, dist in east_wknd_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'East', 'Shift': 'Weekend', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    for worker, zone, prio, dist in west_wknd_assign.get(date, []): 
                        assignments_on_date.append({'Split': 'West', 'Shift': 'Weekend', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                    
                    if assignments_on_date:
                        df_assignments = pd.DataFrame(assignments_on_date)
                        df_assignments = df_assignments.sort_values(by=['Split', 'Shift', 'Worker']).reset_index(drop=True)
                        st.dataframe(df_assignments)
                    else:
                        st.write("No assignments for this day.")