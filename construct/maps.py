import requests
import argparse
import sys
import random
from typing import List
from dataclasses import dataclass

from config import GOOGLE_MAPS_API_KEY

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"

MAX_WAYPOINTS = 20

@dataclass
class Location:
    lat: float
    long: float

@dataclass
class Step:
    start_point: Location
    end_point: Location
    html_instructions: str
    distance: int # ! distance is going to be step['distance']['value']

@dataclass
class RouteInfo:
    origin: Location
    destination: Location
    waypoints: list[Location]

def get_danger_score(location: Location) -> float:
    """
    Right now, randomly score between 0 and 1.
    
    TODO: Eventually pull this function from the 'score.py' file in the 'score' module. 
    """
    return random.random()

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

# def reverse_geocode_best_effort(api_key: str, lat: float, lng: float) -> dict | None:
#     """
#     1) Try result_type=intersection
#     2) Fallback: plain reverse geocode; accept route/street_address/locality etc.
#     Returns the first result (dict) or None.
#     """
#     params = {"latlng": f"{lat},{lng}", "key": api_key, "result_type": "intersection"}
#     r = requests.get(GEOCODE_URL, params=params, timeout=15)
#     data = r.json()
#     if data.get("status") == "OK" and data.get("results"):
#         return data["results"][0]

#     # Fallback: no filter
#     params = {"latlng": f"{lat},{lng}", "key": api_key}
#     r = requests.get(GEOCODE_URL, params=params, timeout=15)
#     data = r.json()
#     if data.get("status") == "OK" and data.get("results"):
#         return data["results"][0]
#     return None

# def neighboring_candidates(api_key: str, center_lat: float, center_lng: float, # NEED FALLBACK TO JUST CHOOSE THIS POINT TODO
#                            radii_m: list[float] = [35.0, 80.0],
#                            bearings_deg: int = 16,
#                            want: int = 4) -> list[dict]:
#     """
#     Sample a ring of points around (center_lat,center_lng), reverse-geocode with
#     best-effort (intersection first, else route/street), dedupe by place_id+name,
#     sort by distance, and return up to `want` candidates.
#     """
#     found, seen = [], set()
#     for radius in radii_m:
#         for k in range(bearings_deg):
#             theta = 2*math.pi * k / bearings_deg
#             dn = radius * math.cos(theta)   # meters north
#             de = radius * math.sin(theta)   # meters east
#             dlat, dlng = meters_to_offset_deg(center_lat, dn, de)
#             plat, plng = center_lat + dlat, center_lng + dlng

#             res = reverse_geocode_best_effort(api_key, plat, plng)
#             if not res:
#                 continue

#             pid = res.get("place_id", "")
#             addr = res.get("formatted_address", "")
#             key = (pid, addr)
#             if key in seen:
#                 continue
#             seen.add(key)

#             loc = res["geometry"]["location"]
#             dist = haversine_m(center_lat, center_lng, loc["lat"], loc["lng"])
#             res["_distance_m"] = dist
#             # Also store a waypoint string you can pass directly to Directions
#             res["_waypoint"] = f"{loc['lat']:.6f},{loc['lng']:.6f}"
#             found.append(res)

#     found.sort(key=lambda r: r.get("_distance_m", 1e9))
#     return found[:want]


class StepProcessor:
    def __init__(self, key: str, origin: str, destination: str, waypoints: list[str] = []):
        self.key = key
        self.origin = origin
        self.destination = destination
        self.waypoints = waypoints
        self.locations = {}
        self.steps = []

    def init_walking_route(self):
        data = get_walking_route(api_key=self.key,
                                 origin=self.origin,
                                 destination=self.destination,
                                 waypoints=self.waypoints)

        steps_json = data['routes'][0]['legs'][0]['steps']
        if steps_json is None:
            raise ValueError("The steps json is empty - reinitialize your start params to fix this.")

        for step in steps_json:
            start_location = Location(lat=step['start_location']['lat'], long=step['start_location']['long'])
            end_location = Location(lat=step['end_location']['lat'], long=step['end_location']['long'])
            self.steps.append(Step(start_point=start_location,
                                   end_point=end_location,
                                   html_instructions=step['html_instructions'],
                                   distance=step['distance']['value']))
            if start_location not in self.locations:
                self.locations[start_location] = step['html_instructions']
            if end_location not in self.locations:
                self.locations[end_location] = step['html_instructions']
    
    def init_steps_and_locations(self):
        pass

    def get_candidate_waypoints(self, location: Location):
        pass

    def get_ui_info(self):
        """
        get_ui_info:

        this function outputs the RouteInfo object that includes the updated waypoints.
        """
        route_info = RouteInfo(origin=self.origin,
                               destination=self.destination,
                               waypoints=self.waypoints)
        
        return route_info

    def add_candidate_waypoints(self):
        """
        add_dangerous_waypoints:

        In this function, we add a candidate waypoint if one of the locations in our route are 
        """
        for location in self.locations:
            if get_danger_score(location) > 0.5:
                # get_candidate_waypoint
                self.waypoint.append(location) # ? appending a location if the danger score > 0.5

    
    def get_top_danger_locations(self):
        """
        top_danger_locations: we want to get the top MAX_WAYPOINTS number of danger locations.
        """
        pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("origin", help="Address or 'lat,lng'")
    parser.add_argument("destination", help="Address or 'lat,lng'")
    parser.add_argument("--key", default=GOOGLE_MAPS_API_KEY, help="Google API key")
    parser.add_argument("--probe_m", type=float, default=25.0, help="Probe radius (meters) for neighbor search")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of replanning iterations")
    parser.add_argument("--max_dist", type=float, default=100.0, help="Maximum distance to walk between evaluations")

    args = parser.parse_args()

    step_processor = StepProcessor(args.key, args.origin, args.destination, [])
    step_processor.init_walking_route()
    print(step_processor.destination)


if __name__ == "__main__":
    main()







