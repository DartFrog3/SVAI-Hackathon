# utils.py
from typing import Tuple, Dict, Any, Optional
import os
import requests
from maps import StepProcessor

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
    Generate safest route.
    """
    StepProcessor(os.getenv("GOOGLE_MAPS_API_KEY"), origin, destination)
    step_processor.reroute_until_safe()
    return step_processor.get_ui_info()
