#!/usr/bin/env python3
import argparse, json, os, re, sys, math
from typing import List, Tuple, Dict
import requests

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"

# ----------------------------- Utilities -----------------------------

def strip_html(s: str) -> str:
    return re.sub("<.*?>", "", s or "").replace("&nbsp;", " ").strip()

def summarize_and_print(route: dict, title: str = "Route summary"):
    route0 = route["routes"][0]
    legs = route0["legs"]
    total_m = sum(leg["distance"]["value"] for leg in legs)
    total_s = sum(leg["duration"]["value"] for leg in legs)
    print(f"\n=== {title} ===")
    print(f"Total distance: {total_m/1000.0:.2f} km")
    print(f"Estimated time: {int(round(total_s/60.0))} min\n")
    print("Turn-by-turn:")
    step_no = 1
    for leg in legs:
        for step in leg["steps"]:
            primary = strip_html(step.get("html_instructions", ""))
            dist = step.get("distance", {}).get("text", "")
            print(f"{step_no:>2}. {primary}  ({dist})")
            step_no += 1

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def meters_to_offset_deg(lat, d_north_m, d_east_m) -> Tuple[float,float]:
    dlat = d_north_m / 111320.0
    dlng = d_east_m / (111320.0 * math.cos(math.radians(lat)) + 1e-12)
    return dlat, dlng

# ----------------------------- API calls -----------------------------

def get_walking_route(api_key: str, origin: str, destination: str, waypoints: List[str]) -> dict:
    params = {"origin": origin, "destination": destination, "mode": "walking", "key": api_key}
    if waypoints:
        params["waypoints"] = "|".join(waypoints)
    r = requests.get(DIRECTIONS_URL, params=params, timeout=20)
    data = r.json()
    if data.get("status") != "OK":
        print(f"Directions API error: {data.get('status')} {data.get('error_message','')}", file=sys.stderr)
        sys.exit(2)
    return data

def reverse_geocode_intersection(api_key: str, lat: float, lng: float) -> Dict | None:
    params = {"latlng": f"{lat},{lng}", "key": api_key, "result_type": "intersection"}
    r = requests.get(GEOCODE_URL, params=params, timeout=15)
    data = r.json()
    if data.get("status") != "OK":
        return None
    results = data.get("results", [])
    return results[0] if results else None

# ----------------------------- First block & neighbors -----------------------------

def first_block_midpoint(route_json: dict) -> Tuple[float,float]:
    legs = route_json["routes"][0]["legs"]
    if not legs or not legs[0]["steps"]:
        raise RuntimeError("No steps in route.")
    step0 = legs[0]["steps"][0]
    a = step0["start_location"]["lat"], step0["start_location"]["lng"]
    b = step0["end_location"]["lat"],   step0["end_location"]["lng"]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

def neighboring_intersections(api_key: str, center_lat: float, center_lng: float,
                              probe_m: float = 25.0, want: int = 4) -> List[Dict]:
    bearings = [(probe_m, 0.0), (0.0, probe_m), (-probe_m, 0.0), (0.0, -probe_m)]
    seen = {}
    found = []
    for dn, de in bearings:
        dlat, dlng = meters_to_offset_deg(center_lat, dn, de)
        plat, plng = center_lat + dlat, center_lng + dlng
        res = reverse_geocode_intersection(api_key, plat, plng)
        if not res:
            continue
        pid = res.get("place_id")
        loc = res["geometry"]["location"]
        dist = haversine_m(center_lat, center_lng, loc["lat"], loc["lng"])
        if pid and pid not in seen:
            seen[pid] = True
            res["_distance_m"] = dist
            found.append(res)
    found.sort(key=lambda r: r.get("_distance_m", 1e9))
    return found[:want]

# ----------------------------- Scoring stub -----------------------------

def score_neighbors(neighbors: List[Dict]) -> Dict | None:
    """
    Placeholder scoring function.
    Input: list of reverse-geocoded intersection results.
    Output: ONE selected neighbor (dict) or None.
    ---------------------------------------------------------------------
    TODO: Replace with your scoring logic, e.g.:
      - prefer smallest _distance_m
      - prefer certain street names
      - prefer intersections closer to the original destination
      - ML model score, etc.
    For now, we pick the nearest by _distance_m.
    """
    if not neighbors:
        return None
    # Nearest by distance (already sorted), but be explicit:
    return min(neighbors, key=lambda r: r.get("_distance_m", 1e9))

# ----------------------------- Main -----------------------------

def main():
    p = argparse.ArgumentParser(description="Walking route -> pick neighbor -> re-route via selected waypoint")
    p.add_argument("origin", help="Address or 'lat,lng'")
    p.add_argument("destination", help="Address or 'lat,lng'")
    p.add_argument("--waypoint", action="append", default=[], help="Optional pre-set waypoint(s)")
    p.add_argument("--key", default=os.getenv("GOOGLE_MAPS_API_KEY"), help="Google API key")
    p.add_argument("--probe_m", type=float, default=25.0, help="Probe radius (meters) for neighbor search")
    args = p.parse_args()

    if not args.key:
        print("Provide API key via --key or GOOGLE_MAPS_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # 1) Initial walking route
    base_route = get_walking_route(args.key, args.origin, args.destination, args.waypoint)
    summarize_and_print(base_route, title="Base walking route")

    # 2) First-block midpoint
    lat, lng = first_block_midpoint(base_route)
    print(f"\nFirst block midpoint: {lat:.6f}, {lng:.6f}")

    # 3) Neighboring intersections (up to 4)
    neighbors = neighboring_intersections(args.key, lat, lng, probe_m=args.probe_m, want=4)
    if not neighbors:
        print("No neighboring intersections found. Try increasing --probe_m to 40–60.")
        return

    print("\nCandidate neighboring intersections:")
    for i, res in enumerate(neighbors, 1):
        loc = res["geometry"]["location"]
        print(f"{i}. {res.get('formatted_address','(no address)')} "
              f"[{loc['lat']:.6f},{loc['lng']:.6f}]  ~{res.get('_distance_m', float('nan')):.0f} m")

    # 4) Score & select one neighbor
    selected = score_neighbors(neighbors)
    if not selected:
        print("Scoring returned no selection; skipping re-route.")
        return

    sel_loc = selected["geometry"]["location"]
    sel_wp = f"{sel_loc['lat']:.6f},{sel_loc['lng']:.6f}"
    print(f"\nSelected neighbor (via scoring): {selected.get('formatted_address','(no address)')}  -> {sel_wp}")

    # 5) Re-call Directions with the selected neighbor as a new waypoint
    new_waypoints = list(args.waypoint) + [sel_wp]
    routed = get_walking_route(args.key, args.origin, args.destination, new_waypoints)
    summarize_and_print(routed, title="Re-routed via selected intersection")

if __name__ == "__main__":
    main()
