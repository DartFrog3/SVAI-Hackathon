
import requests
import argparse
import sys
import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import GOOGLE_MAPS_API_KEY

from score import get_danger_score_from_loc

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"

MAX_WAYPOINTS = 20

# Thresholds
DETECT_THRESHOLD = 0.5   # points above this are considered "dangerous" and trigger a search for alternatives
DANGER_THRESHOLD = 0.7   # acceptable upper bound for a candidate waypoint's danger

@dataclass(frozen=True)
class Location:
    lat: float
    lng: float

@dataclass
class Step:
    start_point: Location
    end_point: Location
    html_instructions: str
    distance: int  # meters

@dataclass
class RouteInfo:
    origin: str
    destination: str
    waypoints: List[str]

# -------------------------
# Utilities
# -------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

def meters_to_offset_deg(lat_ref: float, dn_m: float, de_m: float) -> Tuple[float, float]:
    """Approx convert meters north/east to degrees lat/lng near lat_ref."""
    dlat = dn_m / 111320.0
    dlng = de_m / (111320.0 * math.cos(math.radians(lat_ref)) + 1e-12)
    return dlat, dlng

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decodes a polyline that was encoded using the Google Encoded Polyline Algorithm Format."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'lat': 0, 'lng': 0}
    while index < len(polyline_str):
        for unit in ['lat', 'lng']:
            shift, result = 0, 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)
        lat += changes['lat']
        lng += changes['lng']
        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates

def resample_polyline(points: List[Tuple[float, float]], spacing_m: float) -> List[Location]:
    """Return points along polyline spaced ~every spacing_m (including start/end)."""
    if not points:
        return []
    samples = [Location(points[0][0], points[0][1])]
    acc = 0.0
    for i in range(1, len(points)):
        a = points[i-1]
        b = points[i]
        seg_len = haversine_m(a[0], a[1], b[0], b[1])
        if seg_len == 0:
            continue
        while acc + seg_len >= spacing_m:
            ratio = (spacing_m - acc) / seg_len
            lat = a[0] + ratio * (b[0] - a[0])
            lng = a[1] + ratio * (b[1] - a[1])
            samples.append(Location(lat, lng))
            # reset accumulator relative to the new point
            a = (lat, lng)
            seg_len = haversine_m(a[0], a[1], b[0], b[1])
            acc = 0.0
        acc += seg_len
    # ensure we include the final point
    end = points[-1]
    if samples[-1].lat != end[0] or samples[-1].lng != end[1]:
        samples.append(Location(end[0], end[1]))
    return samples

# -------------------------
# Scoring
# -------------------------
def get_danger_score_old(location: Location) -> float:
    """
    Placeholder scoring function. Returns a random score in [0, 1).
    In production, import from your 'score' module:
        from score import get_danger_score
    """
    return random.random()

def get_danger_score(location: Location) -> float:
    return get_danger_score_from_loc(location)

# -------------------------
# Google API helpers
# -------------------------
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

def reverse_geocode_best_effort(api_key: str, lat: float, lng: float) -> Optional[dict]:
    """
    1) Try result_type=intersection
    2) Fallback: plain reverse geocode; accept route/street_address/locality etc.
    Returns the first result (dict) or None.
    """
    params = {"latlng": f"{lat},{lng}", "key": api_key, "result_type": "intersection"}
    r = requests.get(GEOCODE_URL, params=params, timeout=15)
    data = r.json()
    if data.get("status") == "OK" and data.get("results"):
        return data["results"][0]

    # Fallback: no filter
    params = {"latlng": f"{lat},{lng}", "key": api_key}
    r = requests.get(GEOCODE_URL, params=params, timeout=15)
    data = r.json()
    if data.get("status") == "OK" and data.get("results"):
        return data["results"][0]
    return None

def neighboring_candidates(api_key: str,
                           center_lat: float,
                           center_lng: float,
                           radii_m: List[float] = [35.0, 80.0],
                           bearings_deg: int = 16,
                           want: int = 6) -> List[dict]:
    """
    Sample a ring of points around (center_lat,center_lng), reverse-geocode with
    best-effort (intersection first, else route/street), dedupe by place_id+name,
    sort by distance, and return up to `want` candidates.
    """
    found, seen = [], set()
    for radius in radii_m:
        for k in range(bearings_deg):
            theta = 2 * math.pi * k / bearings_deg
            dn = radius * math.cos(theta)   # meters north
            de = radius * math.sin(theta)   # meters east
            dlat, dlng = meters_to_offset_deg(center_lat, dn, de)
            plat, plng = center_lat + dlat, center_lng + dlng

            res = reverse_geocode_best_effort(api_key, plat, plng)
            if not res:
                continue

            pid = res.get("place_id", "")
            addr = res.get("formatted_address", "")
            key = (pid, addr)
            if key in seen:
                continue
            seen.add(key)

            loc = res["geometry"]["location"]
            dist = haversine_m(center_lat, center_lng, loc["lat"], loc["lng"])
            res["_distance_m"] = dist
            # Also store a waypoint string you can pass directly to Directions
            res["_waypoint"] = addr if addr else f"{loc['lat']:.6f},{loc['lng']:.6f}"
            found.append(res)

    found.sort(key=lambda r: r.get("_distance_m", 1e9))
    return found[:want]

def first_block_midpoint(route_json: dict) -> Tuple[float, float]:
    legs = route_json["routes"][0]["legs"]
    if not legs or not legs[0]["steps"]:
        raise RuntimeError("No steps in route.")
    step0 = legs[0]["steps"][0]
    a = step0["start_location"]["lat"], step0["start_location"]["lng"]
    b = step0["end_location"]["lat"],   step0["end_location"]["lng"]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

# -------------------------
# Core processor
# -------------------------
class StepProcessor:
    def __init__(self,
                 key: str,
                 origin: str,
                 destination: str,
                 waypoints: Optional[List[str]] = None,
                 probe_m: float = 25.0,
                 max_iters: int = 100,
                 max_dist: float = 100.0,
                 detect_threshold: float = DETECT_THRESHOLD,
                 danger_threshold: float = DANGER_THRESHOLD):
        self.key = key
        self.origin = origin
        self.destination = destination
        self.waypoints: List[str] = list(waypoints or [])
        self.probe_m = probe_m
        self.max_iters = max_iters
        self.max_dist = max_dist
        self.detect_threshold = detect_threshold
        self.danger_threshold = danger_threshold

        self.steps: List[Step] = []
        self.sampled_locations: List[Location] = []
        self.scores: dict[Location, float] = {}
        self._baseline_stats = None
        self._final_stats = None

    # ------------ Public API
    def reroute_until_safe(self):
        """
        Iteratively:
          - Fetch route for current (origin, destination, waypoints)
          - Sample locations along steps at most `max_dist` apart
          - Score all locations
          - Find the most dangerous point (> detect_threshold)
          - Generate nearby intersection candidates and pick the best with score < danger_threshold (or lowest)
          - Add it as a waypoint (address if possible)
          - Repeat until no dangerous points remain, MAX_WAYPOINTS reached, or max_iters exhausted.
        """
        # Baseline pass
        self._analyze_current_route()
        self._baseline_stats = self._compute_stats(self.scores)

        iters = 0
        while iters < self.max_iters and len(self.waypoints) < MAX_WAYPOINTS:
            iters += 1
            # Identify top dangerous locations
            top_locs = self.get_top_danger_locations(limit=MAX_WAYPOINTS - len(self.waypoints))
            if not top_locs:
                break

            # Add only one waypoint per iteration, then recompute route
            chosen_loc = top_locs[0]
            wp = self.get_candidate_waypoint(chosen_loc)
            if not wp:
                # If we failed to get a candidate, try the next one this iteration.
                # If none succeed, stop.
                added = False
                for loc in top_locs[1:]:
                    wp = self.get_candidate_waypoint(loc)
                    if wp:
                        self.waypoints.append(wp)
                        added = True
                        break
                if not added:
                    break
            else:
                self.waypoints.append(wp)

            # Re-analyze route after adding a waypoint
            self._analyze_current_route()

        self._final_stats = self._compute_stats(self.scores)

    def get_ui_info(self) -> RouteInfo:
        return RouteInfo(origin=self.origin, destination=self.destination, waypoints=self.waypoints)

    def print_stats(self):
        def fmt(stats):
            return (
                f"count={stats['count']}  "
                f"danger(>={self.detect_threshold:.2f})={stats['num_danger']}  "
                f"mean={stats['mean']:.3f}  "
                f"max={stats['max']:.3f}"
            )

        print("== Safe Routing Stats ==")
        if self._baseline_stats:
            print("Before:", fmt(self._baseline_stats))
        if self._final_stats:
            print("After: ", fmt(self._final_stats))
        if self._baseline_stats and self._final_stats:
            delta = self._baseline_stats['num_danger'] - self._final_stats['num_danger']
            print(f"Waypoints used: {len(self.waypoints)} (MAX {MAX_WAYPOINTS})")
            print(f"Danger point reduction: {delta} (from {self._baseline_stats['num_danger']} to {self._final_stats['num_danger']})")

    # ------------ Internals
    def _analyze_current_route(self):
        data = get_walking_route(api_key=self.key,
                                 origin=self.origin,
                                 destination=self.destination,
                                 waypoints=self.waypoints)
        steps_json = data['routes'][0]['legs'][0]['steps']
        if not steps_json:
            raise ValueError("The steps JSON is empty - reinitialize your start params to fix this.")

        # Extract step objects
        self.steps = []
        self.sampled_locations = []
        self.scores = {}

        for step in steps_json:
            start_location = Location(lat=step['start_location']['lat'], lng=step['start_location']['lng'])
            end_location   = Location(lat=step['end_location']['lat'],   lng=step['end_location']['lng'])
            self.steps.append(Step(start_point=start_location,
                                   end_point=end_location,
                                   html_instructions=step.get('html_instructions', ''),
                                   distance=int(step['distance']['value'])))

            # Sample along this step
            poly = []
            if 'polyline' in step and 'points' in step['polyline']:
                poly = decode_polyline(step['polyline']['points'])
            else:
                # Fallback to just start/end
                poly = [(start_location.lat, start_location.lng), (end_location.lat, end_location.lng)]

            step_samples = resample_polyline(poly, max(self.max_dist, 1.0))
            self.sampled_locations.extend(step_samples)

        # Deduplicate sampled points by rounding to ~6 decimal places (~0.1m-1m)
        dedup = {}
        for loc in self.sampled_locations:
            key = (round(loc.lat, 6), round(loc.lng, 6))
            dedup[key] = loc
        self.sampled_locations = list(dedup.values())

        # Score all sampled locations
        for loc in self.sampled_locations:
            self.scores[loc] = get_danger_score(loc)

    def _compute_stats(self, scores: dict) -> dict:
        vals = list(scores.values())
        if not vals:
            return {"count": 0, "num_danger": 0, "mean": 0.0, "max": 0.0}
        count = len(vals)
        num_danger = sum(1 for v in vals if v >= self.detect_threshold)
        mean = sum(vals) / count
        mx = max(vals)
        return {"count": count, "num_danger": num_danger, "mean": mean, "max": mx}

    def get_top_danger_locations(self, limit: int = 5, min_separation_m: float = 30.0) -> List[Location]:
        """Return up to `limit` most dangerous sampled locations (score >= detect_threshold), spatially filtered."""
        # Sort by score descending
        candidates = [(loc, self.scores[loc]) for loc in self.sampled_locations if self.scores[loc] >= self.detect_threshold]
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected: List[Location] = []
        for loc, _ in candidates:
            too_close = False
            for s in selected:
                if haversine_m(loc.lat, loc.lng, s.lat, s.lng) < min_separation_m:
                    too_close = True
                    break
            if not too_close:
                selected.append(loc)
            if len(selected) >= limit:
                break
        return selected

    def get_candidate_waypoint(self, location: Location) -> Optional[str]:
        """
        For a dangerous location, propose a nearby intersection/address whose danger score is < danger_threshold.
        If none found under the threshold, return the candidate with lowest score.
        Returns a waypoint string suitable for Directions API (address preferred).
        """
        cands = neighboring_candidates(self.key, location.lat, location.lng,
                                       radii_m=[self.probe_m, self.probe_m * 2.0],
                                       bearings_deg=16,
                                       want=6)
        if not cands:
            # As a fallback, try to reverse-geocode the location itself
            rg = reverse_geocode_best_effort(self.key, location.lat, location.lng)
            if rg:
                loc = rg["geometry"]["location"]
                return rg.get("formatted_address", f"{loc['lat']:.6f},{loc['lng']:.6f}")
            return f"{location.lat:.6f},{location.lng:.6f}"

        best_under = None
        best_over = None
        best_over_score = 1e9
        for c in cands:
            loc = c["geometry"]["location"]
            probe = Location(loc["lat"], loc["lng"])
            score = get_danger_score(probe)
            if score < self.danger_threshold:
                # first acceptable under threshold; prefer the closest
                if best_under is None or c["_distance_m"] < best_under["_distance_m"]:
                    best_under = c
            else:
                if score < best_over_score:
                    best_over = c
                    best_over_score = score

        chosen = best_under if best_under is not None else best_over
        if chosen is None:
            return None
        return chosen.get("_waypoint") or chosen.get("formatted_address")


def main():
    parser = argparse.ArgumentParser(description="Construct a safer walking route by inserting waypoints around dangerous areas.")
    parser.add_argument("origin", help="Address or 'lat,lng'")
    parser.add_argument("destination", help="Address or 'lat,lng'")
    parser.add_argument("--key", default=GOOGLE_MAPS_API_KEY, help="Google API key")
    parser.add_argument("--probe_m", type=float, default=25.0, help="Probe radius (meters) for neighbor search")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of replanning iterations")
    parser.add_argument("--max_dist", type=float, default=100.0, help="Sample spacing along the route in meters")
    parser.add_argument("--detect_threshold", type=float, default=DETECT_THRESHOLD, help="Score threshold to flag a location as dangerous")
    parser.add_argument("--danger_threshold", type=float, default=DANGER_THRESHOLD, help="Upper bound score for acceptable candidate waypoints")

    args = parser.parse_args()

    step_processor = StepProcessor(args.key,
                                   args.origin,
                                   args.destination,
                                   [],
                                   probe_m=args.probe_m,
                                   max_iters=args.max_iters,
                                   max_dist=args.max_dist,
                                   detect_threshold=args.detect_threshold,
                                   danger_threshold=args.danger_threshold)
    step_processor.reroute_until_safe()
    info = step_processor.get_ui_info()

    # Output
    print("Destination:", info.destination)
    print("Origin     :", info.origin)
    print(f"Waypoints  : {len(info.waypoints)}")
    for i, w in enumerate(info.waypoints, 1):
        print(f"  {i:02d}. {w}")

    step_processor.print_stats()


if __name__ == "__main__":
    main()
