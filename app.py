# app.py
import os
import streamlit as st
import polyline
import folium
from streamlit_folium import st_folium
from typing import Optional, Dict, Any, List, Tuple
from utils import process_inputs, get_route

st.set_page_config(page_title="Route Viewer", page_icon="🗺️", layout="centered")

# --- Session state init ---
if "coords" not in st.session_state:
    st.session_state.coords = None
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "route_meta" not in st.session_state:
    st.session_state.route_meta = None

# --- Helpers ---
def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets; fallback to environment
    try:
        return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_MAPS_API_KEY")

def decode_route_points(directions: Dict[str, Any]) -> Tuple[List[tuple], Dict[str, Any]]:
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
st.title("🗺️ Safe Route Generation")
st.caption("Enter an origin and destination. We’ll find you the safest way home.")

# --- Form to avoid flicker and persist submission ---
with st.form("route_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origin", placeholder="e.g., Golden Gate Bridge, San Francisco, CA")
    with col2:
        destination = st.text_input("Destination", placeholder="e.g., Ferry Building, San Francisco, CA")
    mode = st.selectbox("Travel mode", ["driving", "walking", "bicycling", "transit"], index=0)
    submitted = st.form_submit_button("Get Route", use_container_width=True)

# --- On submit: get route and SAVE to session_state ---
if submitted:
    api_key = get_api_key()
    if not api_key:
        st.error("No API key found. Add to `.streamlit/secrets.toml` or set `GOOGLE_MAPS_API_KEY`.")
    elif not origin or not destination:
        st.warning("Please provide both origin and destination.")
    else:
        o, d = process_inputs(origin, destination)
        with st.spinner("Fetching route via dummy function..."):
            directions_raw = get_route(o, d, mode=mode, api_key=api_key)

        status = directions_raw.get("status", "UNKNOWN")
        if status != "OK" or not directions_raw.get("routes"):
            st.error(f"Directions provider returned status: {status}")
            msg = directions_raw.get("error_message")
            if msg:
                st.caption(msg)
        else:
            coords, _ = decode_route_points(directions_raw)
            st.session_state.coords = coords
            st.session_state.route_data = directions_raw
            st.session_state.route_meta = {"origin": o, "destination": d, "mode": mode}

# --- ALWAYS render from session_state if we have a route ---
if st.session_state.coords and st.session_state.route_data and st.session_state.route_meta:
    coords = st.session_state.coords
    meta = st.session_state.route_meta
    directions = st.session_state.route_data

    # Draw map
    draw_map(coords, meta["origin"], meta["destination"])

    # Details
    route = directions["routes"][0]
    leg0 = route["legs"][0]
    total_distance = leg0["distance"]["text"]
    total_duration = leg0["duration"]["text"]

    st.success(f"Route found: **{total_distance}**, about **{total_duration}** ({meta['mode']}).")

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

    # Optional: clear button
    if st.button("Clear route"):
        st.session_state.coords = None
        st.session_state.route_data = None
        st.session_state.route_meta = None
