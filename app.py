# app.py
import os
import streamlit as st
import polyline
import folium
from streamlit_folium import st_folium
from typing import Optional, Tuple, Dict, Any, List
from utils import process_inputs, get_route

st.set_page_config(page_title="Route Viewer", page_icon="🗺️", layout="centered")

# --- Helpers ---
def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets; fallback to environment
    key = None
    try:
        key = st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        key = os.getenv("GOOGLE_MAPS_API_KEY")
    return key

def decode_route_points(directions: Dict[str, Any]):
    route = directions["routes"][0]
    enc = route["overview_polyline"]["points"]
    coords = polyline.decode(enc)  # [(lat, lng), ...]
    return coords, route

def draw_map(coords: List[tuple], origin_label: str, destination_label: str) -> None:
    if not coords:
        st.warning("No coordinates to draw.")
        return
    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, weight=6).add_to(m)
    folium.Marker(coords[0], tooltip=f"Origin: {origin_label}").add_to(m)
    folium.Marker(coords[-1], tooltip=f"Destination: {destination_label}").add_to(m)
    st_folium(m, width=900, height=550)

# --- UI ---
st.title("🗺️ Google Maps Route Viewer")
st.caption("Enter an origin and destination. We’ll pass them through a dummy function that returns a Google Maps Directions response, and we render the route.")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Travel mode", ["driving", "walking", "bicycling", "transit"], index=0)
    st.markdown("**API key** is read from `.streamlit/secrets.toml` or `GOOGLE_MAPS_API_KEY` env var.")
    if not get_api_key():
        st.error("No API key found. Add to `.streamlit/secrets.toml` or set `GOOGLE_MAPS_API_KEY`.")

col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("Origin", placeholder="e.g., Golden Gate Bridge, San Francisco, CA")
with col2:
    destination = st.text_input("Destination", placeholder="e.g., Ferry Building, San Francisco, CA")

go = st.button("Get Route", type="primary", use_container_width=True)

if go:
    api_key = get_api_key()
    if not api_key:
        st.stop()

    if not origin or not destination:
        st.warning("Please provide both origin and destination.")
        st.stop()

    # Pass through your dummy preprocessor
    origin_proc, destination_proc = process_inputs(origin, destination)

    with st.spinner("Fetching route via dummy function..."):
        directions_raw = get_route(origin_proc, destination_proc, mode=mode, api_key=api_key)

    # Basic response checks
    status = directions_raw.get("status", "UNKNOWN")
    if status != "OK" or not directions_raw.get("routes"):
        st.error(f"Directions provider returned status: {status}")
        msg = directions_raw.get("error_message")
        if msg:
            st.caption(msg)
        st.stop()

    # Decode + draw
    coords, route = decode_route_points(directions_raw)
    draw_map(coords, origin_proc, destination_proc)

    # Details panel
    leg0 = route["legs"][0]
    total_distance = leg0["distance"]["text"]
    total_duration = leg0["duration"]["text"]

    st.success(f"Route found: **{total_distance}**, about **{total_duration}** ({mode}).")

    with st.expander("Route details (first leg)"):
        st.write(f"**Start address:** {leg0.get('start_address','')}")
        st.write(f"**End address:** {leg0.get('end_address','')}")
        st.write(f"**Distance:** {total_distance}")
        st.write(f"**Duration:** {total_duration}")

        if leg0.get("steps"):
            step0 = leg0["steps"][0]
            st.write("**First step:**")
            st.json({
                "distance_text": step0["distance"]["text"],
                "end_location": step0["end_location"],
                "html_instructions": step0.get("html_instructions", "")
            })

        show_steps = st.checkbox("Show all step instructions (HTML from API)")
        if show_steps:
            for i, step in enumerate(leg0.get("steps", []), start=1):
                st.markdown(f"**Step {i}:** {step.get('html_instructions','')}", unsafe_allow_html=True)
                st.caption(f"{step['distance']['text']} • {step['duration']['text']}")
