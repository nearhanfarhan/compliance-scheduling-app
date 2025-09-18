import streamlit as st
from streamlit_oauth import OAuth2Component
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
# --- NEW IMPORTS FOR DECODING GOOGLE'S TOKEN ---
from google.oauth2 import id_token
from google.auth.transport import requests as goog_requests


# =======================================
# App Configuration & Secrets
# =======================================

st.set_page_config(page_title="Cyclist Team Scheduler", layout="wide")
st.title("🚴 London Compliance Team Scheduling Tool")

# Use Streamlit's secrets management for credentials and API keys
try:
    GOOGLE_MAPS_API_KEY = st.secrets["api_keys"]["google_maps"]
    SERVICE_ACCOUNT_CREDS = st.secrets["google_credentials"]
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    CLIENT_ID = st.secrets["oauth_credentials"]["client_id"]
    CLIENT_SECRET = st.secrets["oauth_credentials"]["client_secret"]
except (KeyError, FileNotFoundError):
    st.error("🚨 Critical Error: Secrets for Google APIs or OAuth credentials not found. Please configure your `secrets.toml` file.")
    st.stop()

# =======================================
# OAuth2 Authentication Setup
# =======================================

AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)

# =======================================
# Main Application Logic
# =======================================

if 'token' not in st.session_state:
    result = oauth2.authorize_button(
        name="Login with Google",
        icon="https://www.google.com.tw/favicon.ico",
        redirect_uri="https://compliance-scheduling.streamlit.app/", # Make sure this is your app's URL
        scope="openid email profile",
        key="google",
        use_container_width=True
    )
    if result:
        st.session_state.token = result.get('token')
        st.rerun()
else:
    token_dict = st.session_state['token']
    id_token_str = token_dict.get('id_token')
    
    try:
        # Verify and decode the ID token
        user_info = id_token.verify_oauth2_token(
            id_token_str, goog_requests.Request(), CLIENT_ID
        )
        user_email = user_info.get('email')

        # Check if the email domain is correct
        if user_email and user_email.endswith(('@li.me', '@ext.li.me')):
            
            st.sidebar.success(f"Logged in as {user_info.get('name')}")
            
            # --- ALL YOUR ORIGINAL APP CODE GOES HERE ---

            # =======================================
            # REFACTORED SCRIPT FUNCTIONS
            # =======================================
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
            
            # ... (All your other functions: generate_weekly_availability, etc. remain here unchanged)
            def generate_weekly_availability(daily_roster_files, employee_info_file, client, cyclist_sheet_id):
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
                for file in daily_roster_files:
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
                try:
                    spreadsheet = client.open_by_key(cyclist_sheet_id)
                    worksheet = spreadsheet.worksheet('Cyclist Details')
                    full_cyclist_df = get_as_dataframe(worksheet).dropna(how="all")[['Email', 'Post Code', 'Split']]
                except Exception as e:
                    st.error(f"Failed to fetch cyclist details from Google Sheets: {e}")
                    return None
                weekly_schedule = pd.merge(employee_df, full_schedule, on='Name', how='right')
                availability_data = pd.merge(weekly_schedule, full_cyclist_df, on='Email', how='left')
                cols = availability_data.columns.tolist()
                if 'Email' in cols:
                  email_idx = cols.index('Email')
                  new_order = cols[:email_idx+1] + ['Post Code', 'Split'] + [col for col in cols[email_idx+1:] if col not in ['Post Code', 'Split']]
                  availability_data = availability_data[new_order]
                return availability_data
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
                            for date in df.columns[5:] 
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
                    for priority in priority_order:
                        current_priority_zones = zones.get(priority, [])
                        if not remaining_workers or not current_priority_zones:
                            continue
                        assigned_zone_names = {a[1] for a in assignments_per_day[day]}
                        filtered_zones = [z for z in current_priority_zones if z[0] not in assigned_zone_names]
                        if not filtered_zones:
                            continue
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
                    for email in remaining_workers:
                        assignments_per_day[day].append((availability[email]["name"], "NOT ASSIGNED", "N/A", 0.0))
                return assignments_per_day
            def build_final_schedule_df(*assignments_list):
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
            def write_schedule_to_google_sheet(client, sheet_id, availability_df, all_assignment_dicts):
                try:
                    spreadsheet = client.open_by_key(sheet_id)
                    sheet_name = "New Schedule"
                    try:
                        worksheet = spreadsheet.worksheet(sheet_name)
                        spreadsheet.del_worksheet(worksheet)
                    except gspread.exceptions.WorksheetNotFound:
                        pass
                    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="30")
                    all_dates = sorted(list(set(date for d in all_assignment_dicts.values() for date in d.keys())), key=lambda x: datetime.strptime(x, '%d/%m'))
                    all_workers = sorted(list(set(assign[0] for d in all_assignment_dicts.values() for assigns in d.values() for assign in assigns)))
                    worker_split_map = pd.Series(availability_df.Split.values, index=availability_df.Name).to_dict()
                    schedule_matrix = [["Worker Name", "Split"] + all_dates]
                    for worker in all_workers:
                        split = worker_split_map.get(worker, "N/A")
                        row = [worker, split] + [""] * len(all_dates)
                        for i, date in enumerate(all_dates):
                            day_zone = next((z[1] for z in all_assignment_dicts['east_day'].get(date, []) if z[0] == worker), "") or next((z[1] for z in all_assignment_dicts['west_day'].get(date, []) if z[0] == worker), "")
                            evening_zone = next((z[1] for z in all_assignment_dicts['east_evening'].get(date, []) if z[0] == worker), "") or next((z[1] for z in all_assignment_dicts['west_evening'].get(date, []) if z[0] == worker), "")
                            weekend_zone = next((z[1] for z in all_assignment_dicts['east_weekend'].get(date, []) if z[0] == worker), "") or next((z[1] for z in all_assignment_dicts['west_weekend'].get(date, []) if z[0] == worker), "")
                            all_zones = [z for z in [day_zone, evening_zone, weekend_zone] if z]
                            cell_value = " / ".join(all_zones)
                            row[i + 2] = cell_value
                        schedule_matrix.append(row)
                    worksheet.update("A1", schedule_matrix)
                    return True, None
                except Exception as e:
                    return False, str(e)

            # =======================================
            # STREAMLIT UI AND MAIN EXECUTION
            # =======================================

            st.sidebar.header("⚙️ Configuration")
            st.sidebar.info("Upload files and provide Google Sheet IDs to generate the schedule.")
            
            fcl_sheet_id = st.sidebar.text_input("Full Cyclist List & Pre-Assignment Sheet ID", "1IH24J2EpdjJKEuvqQrZSfKaElq-k77XJyoSuD5jfUIY")
            zone_sheet_id = st.sidebar.text_input("Zone Definitions Sheet ID", "1e6BZP5yWlOHLeks0wDInanKX2U_DA2CC0SbGjjeotZM")
            
            st.sidebar.header("📄 File Uploads")
            employee_info_file = st.sidebar.file_uploader("1. Upload Team Details CSV", type="csv")
            daily_roster_files = st.sidebar.file_uploader("2. Upload Daily Roster CSVs (Select multiple)", type="csv", accept_multiple_files=True)
            
            if st.button("🚀 Generate Weekly Schedule"):
                if not employee_info_file or not daily_roster_files:
                    st.warning("⚠️ Please upload all required files before generating the schedule.")
                else:
                    with st.spinner("Processing... This might take a few moments."):
                        client = get_gspread_client()
                        st.write("### Step 1: Generating Worker Availability...")
                        availability_df = generate_weekly_availability(daily_roster_files, employee_info_file, client, fcl_sheet_id)
                        if availability_df is None:
                            st.error("Failed to generate availability. Please check the uploaded files and Sheet IDs.")
                            st.stop()
                        st.success("✅ Availability data processed successfully.")
                        with st.expander("View Generated Availability Data"):
                            st.dataframe(availability_df)
                        st.write("### Step 2: Fetching Zone Data and Pre-assignments...")
                        reliable_cyclists_df = fetch_sheet_data(client, fcl_sheet_id, "Cyclist Pre-Assignment")
                        east_day_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_day_zones")
                        east_evening_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_evening_zones")
                        east_weekend_zones_df = fetch_sheet_data(client, zone_sheet_id, "east_weekend_zones")
                        west_day_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_day_zones")
                        west_evening_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_evening_zones")
                        west_weekend_zones_df = fetch_sheet_data(client, zone_sheet_id, "west_weekend_zones")
                        all_dfs = [reliable_cyclists_df, east_day_zones_df, east_evening_zones_df, east_weekend_zones_df, west_day_zones_df, west_evening_zones_df, west_weekend_zones_df]
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
                        st.write("### Step 4: Displaying Final Schedule...")
                        all_assigns = [east_day_assign, east_eve_assign, east_wknd_assign, west_day_assign, west_eve_assign, west_wknd_assign]
                        final_schedule_df = build_final_schedule_df(*all_assigns)
                        st.balloons()
                        st.header("🎉 Final Weekly Schedule 🎉")
                        st.dataframe(final_schedule_df)
                        st.write("### Step 5: Writing to Google Sheets...")
                        all_assignment_dicts = {'east_day': east_day_assign, 'west_day': west_day_assign, 'east_evening': east_eve_assign, 'west_evening': west_eve_assign, 'east_weekend': east_wknd_assign, 'west_weekend': west_wknd_assign}
                        success, error_message = write_schedule_to_google_sheet(client, fcl_sheet_id, availability_df, all_assignment_dicts)
                        if success:
                            st.success("✅ Schedule successfully written to 'New Schedule' tab in your Google Sheet!")
                        else:
                            st.error(f"🚨 Failed to write to Google Sheets: {error_message}")
                        st.header("📋 Daily Assignment Details")
                        all_dates = sorted(list(final_schedule_df.columns[1:]), key=lambda x: datetime.strptime(x, '%d/%m'))
                        for date in all_dates:
                            with st.expander(f"**📅 Assignments for {date}**"):
                                assignments_on_date = []
                                for worker, zone, prio, dist in east_day_assign.get(date, []): assignments_on_date.append({'Split': 'East', 'Shift': 'Day', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                for worker, zone, prio, dist in west_day_assign.get(date, []): assignments_on_date.append({'Split': 'West', 'Shift': 'Day', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                for worker, zone, prio, dist in east_eve_assign.get(date, []): assignments_on_date.append({'Split': 'East', 'Shift': 'Evening', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                for worker, zone, prio, dist in west_eve_assign.get(date, []): assignments_on_date.append({'Split': 'West', 'Shift': 'Evening', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                for worker, zone, prio, dist in east_wknd_assign.get(date, []): assignments_on_date.append({'Split': 'East', 'Shift': 'Weekend', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                for worker, zone, prio, dist in west_wknd_assign.get(date, []): assignments_on_date.append({'Split': 'West', 'Shift': 'Weekend', 'Worker': worker, 'Zone': zone, 'Priority': prio, 'Distance (km)': f"{dist:.2f}"})
                                if assignments_on_date:
                                    df_assignments = pd.DataFrame(assignments_on_date)
                                    df_assignments = df_assignments.sort_values(by=['Split', 'Shift', 'Worker']).reset_index(drop=True)
                                    st.dataframe(df_assignments)
                                else:
                                    st.write("No assignments for this day.")
        else:
            st.error("Access Denied. Please log in with your company email.")
            
    except ValueError as e:
        # Handle invalid or expired token
        st.error(f"Login failed or token is invalid. Please try logging in again. Error: {e}")
        # Clear the faulty token and rerun to show the login button
        if 'token' in st.session_state:
            del st.session_state['token']
        st.rerun()