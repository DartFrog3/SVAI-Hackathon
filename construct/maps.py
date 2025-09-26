import requests
import argparse
import sys
import random
from typing import List
from dataclasses import dataclass

from config import GOOGLE_MAPS_API_KEY

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

@dataclass
class Location:
    lat: float
    long: float
    html_instructions: str


def get_score(location: Location) -> float:
    """
    Right now, randomly score between 0 and 1.
    
    TODO: Eventually pull this function from the 'score.py' file in the 'score' module. 
    """
    return random.random()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("origin", help="Address or 'lat,lng'")
    parser.add_argument("destination", help="Address or 'lat,lng'")
    parser.add_argument("--key", default=GOOGLE_MAPS_API_KEY, help="Google API key")
    parser.add_argument("--probe_m", type=float, default=25.0, help="Probe radius (meters) for neighbor search")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of replanning iterations")
    parser.add_argument("--max_dist", type=float, default=100.0, help="Maximum distance to walk between evaluations")

    args = parser.parse_args()

    # base_route = get_walking_route(args.key, args.origin, args.destination, [])
    # print(base_route['routes'][0]['legs'][0]['steps'][0])


if __name__ == "__main__":
    main()







