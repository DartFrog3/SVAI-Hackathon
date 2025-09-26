# utils.py
from typing import Tuple, Dict, Any, Optional
import os
import requests

def process_inputs(origin: str, destination: str) -> Tuple[str, str]:
    """
    Dummy preprocessor you can customize.
    For now it just strips whitespace.
    """
    return origin.strip(), destination.strip()

def _get_api_key_from_env_or_streamlit() -> Optional[str]:
    """Try pulling key from env; Streamlit will pass via st.secrets in the app layer.
    This helper lets you also run this module standalone in tests by using env var.
    """
    return os.getenv("GOOGLE_MAPS_API_KEY")

def get_route(origin: str, destination: str, mode: str = "driving", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    DUMMY FUNCTION: Provide the route from Google Maps Directions API.
    - Replace or modify this function as you like (e.g., call a different service,
      add caching, enforce constraints, etc.).
    - The app expects a raw Directions API-like response dict containing at least:
      { 'status': 'OK', 'routes': [...] }
    """
    if api_key is None:
        # Fallback for running outside Streamlit; Streamlit passes the key explicitly
        api_key = _get_api_key_from_env_or_streamlit()

    if not api_key:
        return { 'status': 'MISSING_API_KEY', 'routes': [] }

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return { 'status': 'HTTP_ERROR', 'error_message': str(e), 'routes': [] }
    except requests.RequestException as e:
        return { 'status': 'NETWORK_ERROR', 'error_message': str(e), 'routes': [] }
