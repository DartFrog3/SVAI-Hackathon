# Streamlit Google Maps Route Demo

A simple Streamlit app that accepts an origin and destination, runs them through a dummy function (you can customize), calls the Google Directions API, and displays the route on a map.

## Quickstart

1) **Python env**
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **API Key**
- Create `.streamlit/secrets.toml`:
```toml
GOOGLE_MAPS_API_KEY = "YOUR_KEY_HERE"
```
(or set env var `GOOGLE_MAPS_API_KEY`)

3) **Run**
```bash
streamlit run app.py
```

## Notes
- The dummy function lives in `utils.py` (`process_inputs`). Edit it to preprocess, validate, or augment the inputs.
- If you’d like to inspect the first step’s distance and end location, see the “Route details” expander in the UI.
