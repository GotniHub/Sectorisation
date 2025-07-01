import streamlit as st
import pandas as pd
import googlemaps
import requests
import time
from io import BytesIO
 
# Google API Key
# GOOGLE_API_KEY = "AIzaSyBdNUoLkQal9LjqhjzrdXx8Y6_CoS9lmZQ"
GOOGLE_API_KEY = "AIzaSyAGO8U8-I0PqP_x2MJ_ROg82WYCehtA9qs"

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
 
# Streamlit Page Config
st.set_page_config(page_title="Google Business Geocoder", page_icon="📍", layout="wide")
 
# Custom Styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
    }
    .stDownloadButton>button {
        background-color: #28A745;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
    }
    .block-container {
        max-width: 1200px;
    }
    </style>
    """, unsafe_allow_html=True)
 
# Sidebar UI
st.sidebar.header("📌 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel file", type=["xlsx"])
 
# Function to Geocode Address
def geocode_address(address):
    """Geocode an address using Google Maps API."""
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            formatted_address = geocode_result[0]["formatted_address"]
            place_id = geocode_result[0].get("place_id", None)
            return formatted_address, location["lat"], location["lng"], place_id
        else:
            return None, None, None, None
    except Exception:
        return None, None, None, None
 
# Function to Enrich Data Using Google Places API
def enrich_place_details(place_id):
    """Retrieve additional business details using Google Places API."""
    if not place_id:
        return None, None, None, None, None, None
 
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,types,website,formatted_phone_number,rating,user_ratings_total,opening_hours&key={GOOGLE_API_KEY}"
    response = requests.get(url).json()
 
    if "result" in response:
        result = response["result"]
        name = result.get("name", None)
        category = ", ".join(result.get("types", []))
        website = result.get("website", None)
        phone = result.get("formatted_phone_number", None)
        rating = result.get("rating", None)
        reviews = result.get("user_ratings_total", None)
        return name, category, website, phone, rating, reviews
    return None, None, None, None, None, None
 
# Function to Process Geocoding & Enrichment
def process_geocoding(df, selected_columns):
    """Concatenate selected address columns, geocode, and enrich data."""
    df["Full Address"] = df[selected_columns].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
 
    geocoded_data = []
    progress_bar = st.progress(0)
 
    for i, address in enumerate(df["Full Address"]):
        formatted_address, lat, lng, place_id = geocode_address(address)
        name, category, website, phone, rating, reviews = enrich_place_details(place_id)
        geocoded_data.append([formatted_address, lat, lng, place_id, name, category, website, phone, rating, reviews])
       
        time.sleep(0.2)  # Simulate API delay
        progress_bar.progress((i + 1) / len(df))
 
    df[["Formatted Address", "Latitude", "Longitude", "Place ID", "Business Name", "Category", "Website", "Phone", "Rating", "Reviews"]] = pd.DataFrame(geocoded_data)
 
    progress_bar.empty()  # Remove progress bar
    return df
 
# Function to Save Data to Excel
def save_to_excel(df):
    """Save dataframe to an Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched Data")
    processed_file = output.getvalue()
    return processed_file
 
# Main UI
st.title("📍 Google Business API Geocoder & Enrichment")
st.write("Upload an Excel file, select address columns, and enrich data with Google Business API.")
 
if uploaded_file:
    df = pd.read_excel(uploaded_file)
 
    # Show Data Preview Toggle
    if st.sidebar.checkbox("👀 Show Data Preview"):
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())
 
    # Column selection
    st.sidebar.subheader("📌 Select Address Columns")
    selected_columns = st.sidebar.multiselect("Choose at least two columns:", df.columns)
 
    if len(selected_columns) >= 2:
        if st.sidebar.button("🚀 Start Geocoding & Enrichment"):
            with st.spinner("🔍 Processing... This may take a few minutes."):
                df_geocoded = process_geocoding(df, selected_columns)
           
            st.success("✅ Process Completed!")
 
            # Display results
            st.write("### 🎯 Enriched Data Preview:")
            st.dataframe(df_geocoded.head())
 
            # Download Button
            excel_data = save_to_excel(df_geocoded)
            st.download_button(label="📥 Download Enriched File",
                               data=excel_data,
                               file_name="enriched_data.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
            # Show locations on a map
            st.write("### 🌍 Interactive Map of Geocoded Locations:")
            st.map(df_geocoded.dropna(subset=["Latitude", "Longitude"])[["Latitude", "Longitude"]])
 
            # Display enriched business details
            st.write("### 🏢 Business Details Summary")
            for index, row in df_geocoded.head(5).iterrows():
                with st.expander(f"🔹 {row['Business Name']}"):
                    st.write(f"📍 **Address:** {row['Formatted Address']}")
                    st.write(f"🏢 **Category:** {row['Category']}")
                    st.write(f"🌍 **Website:** {row['Website'] if row['Website'] else 'N/A'}")
                    st.write(f"📞 **Phone:** {row['Phone'] if row['Phone'] else 'N/A'}")
                    st.write(f"⭐ **Rating:** {row['Rating']} ({row['Reviews']} Reviews)")
 
    else:
        st.sidebar.warning("⚠️ Please select at least two columns for geocoding.")