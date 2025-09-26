import os
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from newsapi import NewsApiClient
from geopy.distance import geodesic

# State schema for the workflow
class DangerScoreState(dict):
    """State containing location data, news data, crime data, and danger score"""
    latitude: float
    longitude: float
    street_address: str
    news_data: Dict[str, Any]
    crime_data: Dict[str, Any]
    danger_score: float

def crime_agent_node(state: Dict[str, Any], csv_path: str = "crime_data.csv", radius_km: float = 1.0) -> Dict[str, Any]:
    """
    LangGraph node that counts crimes within a preset distance from target location
    """
    try:
        # Load crime data from CSV
        if not os.path.exists(csv_path):
            # Create sample data if CSV doesn't exist (for testing)
            crime_df = create_sample_crime_data()
        else:
            crime_df = pd.read_csv(csv_path)
        
        if crime_df.empty:
            state["crime_data"] = {
                "success": True,
                "crimes_in_radius": 0,
                "total_crimes": 0,
                "crime_ratio": 0.0,
                "radius_km": radius_km,
                "nearby_crimes": []
            }
            return state
        
        # Target location
        target_location = (state["latitude"], state["longitude"])
        
        # Find crimes within radius
        crimes_in_radius = []
        
        for idx, row in crime_df.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
                continue
            
            crime_location = (row['Latitude'], row['Longitude'])
            distance = geodesic(target_location, crime_location).kilometers
            
            if distance <= radius_km:
                crimes_in_radius.append({
                    'intersection': row.get('Intersection', 'Unknown'),
                    'neighborhood': row.get('Analysis Neighborhood', 'Unknown'),
                    'latitude': row['Latitude'],
                    'longitude': row['Longitude'],
                    'distance_km': round(distance, 3)
                })
        
        # Calculate statistics
        total_crimes = len(crime_df)
        crimes_count = len(crimes_in_radius)
        crime_ratio = crimes_count / total_crimes if total_crimes > 0 else 0.0
        
        # Store crime data in state
        state["crime_data"] = {
            "success": True,
            "crimes_in_radius": crimes_count,
            "total_crimes": total_crimes,
            "crime_ratio": crime_ratio,
            "radius_km": radius_km,
            "nearby_crimes": crimes_in_radius[:10],  # Keep top 10 closest
            "search_location": f"{state['latitude']}, {state['longitude']}"
        }
        
    except Exception as e:
        print(f"Error in crime_agent_node: {e}")
        state["crime_data"] = {
            "success": False,
            "error": str(e),
            "crimes_in_radius": 0,
            "total_crimes": 0,
            "crime_ratio": 0.0,
            "radius_km": radius_km,
            "nearby_crimes": []
        }
    
    return state

def create_sample_crime_data() -> pd.DataFrame:
    """Create sample crime data matching the CSV format for testing"""
    import numpy as np
    np.random.seed(42)
    
    # Sample data based on your CSV format
    intersections = [
        "BACHE ST \\ BENTON AVE",
        "48TH AVE \\ GREAT HWY \\ ULLOA ST", 
        "CONKLING ST \\ WATERVILLE ST",
        "16TH ST \\ OWENS ST",
        "MELROSE AVE \\ TERESITA BLVD",
        "BERKSHIRE WAY \\ LAKE MERCED BLVD \\ LAKESHORE DR"
    ]
    
    neighborhoods = [
        "Bernal Heights",
        "Sunset/Parkside", 
        "Bayview Hunters Point",
        "Mission Bay",
        "West of Twin Peaks"
    ]
    
    # Generate sample data around San Francisco area
    base_lat, base_lng = 37.7749, -122.4194
    n_samples = 100
    
    data = {
        'Intersection': np.random.choice(intersections + [f"Street {i} \\ Ave {i}" for i in range(50)], n_samples),
        'Analysis Neighborhood': np.random.choice(neighborhoods, n_samples),
        'Latitude': np.random.normal(base_lat, 0.05, n_samples),  # Spread around SF
        'Longitude': np.random.normal(base_lng, 0.05, n_samples)
    }
    
    return pd.DataFrame(data)

def news_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that fetches news data using NewsApiClient
    """
    try:
        # Initialize NewsAPI client
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            state["news_data"] = {
                "success": False,
                "error": "NEWS_API_KEY not configured",
                "articles": [],
                "crime_related_count": 0,
                "total_articles": 0
            }
            return state
        
        newsapi = NewsApiClient(api_key=api_key)
        
        # Extract city from address for better search
        address_parts = state["street_address"].split(",")
        city = address_parts[1].strip() if len(address_parts) > 1 else address_parts[0].strip()
        
        # Calculate date range (last 30 days)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        
        # Search for general area news
        area_articles = newsapi.get_everything(
            q=f'"{city}"',
            language='en',
            sort_by='publishedAt',
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            page_size=20
        )
        
        # Search for crime-related news in the area
        crime_articles = newsapi.get_everything(
            q=f'"{city}" AND (crime OR police OR arrest OR robbery OR assault OR theft OR shooting OR violence)',
            language='en',
            sort_by='publishedAt',
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            page_size=15
        )
        
        # Combine and analyze articles
        all_articles = area_articles.get('articles', []) + crime_articles.get('articles', [])
        
        # Remove duplicates based on title
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Count crime-related articles
        crime_keywords = ['crime', 'police', 'arrest', 'robbery', 'assault', 'theft', 'shooting', 'violence', 'murder', 'burglary']
        crime_related_count = 0
        
        for article in unique_articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            if any(keyword in content for keyword in crime_keywords):
                crime_related_count += 1
        
        # Store news data in state
        state["news_data"] = {
            "success": True,
            "articles": unique_articles[:10],  # Keep top 10 most relevant
            "total_articles": len(unique_articles),
            "crime_related_count": crime_related_count,
            "search_city": city,
            "date_range": f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}"
        }
        
    except Exception as e:
        print(f"Error in news_agent_node: {e}")
        state["news_data"] = {
            "success": False,
            "error": str(e),
            "articles": [],
            "crime_related_count": 0,
            "total_articles": 0
        }
    
    return state

def danger_score_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that calculates danger score using location, news, and crime data
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
    
    # Prepare news summary
    news_data = state.get("news_data", {})
    if news_data.get("success"):
        news_summary = f"""
Recent news analysis for {news_data.get('search_city', 'the area')}:
- Total articles found: {news_data.get('total_articles', 0)}
- Crime-related articles: {news_data.get('crime_related_count', 0)}
- Time period: {news_data.get('date_range', 'last 30 days')}

Recent headlines:"""
        
        for article in news_data.get('articles', [])[:5]:
            title = article.get('title', 'No title')
            news_summary += f"\n- {title}"
    else:
        news_summary = f"News data unavailable: {news_data.get('error', 'Unknown error')}"
    
    # Prepare crime summary
    crime_data = state.get("crime_data", {})
    if crime_data.get("success"):
        crime_summary = f"""
Crime data analysis:
- Crimes within {crime_data.get('radius_km', 1.0)} km radius: {crime_data.get('crimes_in_radius', 0)}
- Total crimes in dataset: {crime_data.get('total_crimes', 0)}
- Crime density ratio: {crime_data.get('crime_ratio', 0):.3f}

Nearby crime locations:"""
        
        for crime in crime_data.get('nearby_crimes', [])[:5]:
            crime_summary += f"\n- {crime['intersection']} ({crime['neighborhood']}) - {crime['distance_km']}km away"
    else:
        crime_summary = f"Crime data unavailable: {crime_data.get('error', 'Unknown error')}"
    
    prompt = ChatPromptTemplate.from_template("""
    You are a safety expert. Rate the danger level of this location on a scale of 0.0 to 1.0:
    
    Location: {street_address}
    Coordinates: {latitude}, {longitude}
    
    News Analysis:
    {news_summary}
    
    Crime Data Analysis:
    {crime_summary}
    
    Consider factors like:
    - Recent crime incidents from news reports
    - Historical crime density in the immediate area
    - Proximity to known crime locations
    - Neighborhood safety patterns
    - News sentiment and frequency of safety incidents
    
    Respond with ONLY a number between 0.0 and 1.0 where:
    0.0 = completely safe
    1.0 = extremely dangerous
    
    Your response must be a single decimal number with no other text.
    """)
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "street_address": state["street_address"],
            "latitude": state["latitude"],
            "longitude": state["longitude"],
            "news_summary": news_summary,
            "crime_summary": crime_summary
        })
        
        # Extract score from response
        score_text = response.content.strip()
        score = float(score_text)
        
        # Ensure score is in valid range
        state["danger_score"] = max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Error in danger_score_node: {e}")
        state["danger_score"] = 0.5  # Default middle score if error
    
    return state

def create_danger_score_workflow(csv_path: str = "crime_data.csv", radius_km: float = 1.0) -> StateGraph:
    """
    Create a LangGraph workflow with news agent, crime agent, and danger score nodes
    """
    # Create workflow
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("crime_agent", lambda state: crime_agent_node(state, csv_path, radius_km))
    workflow.add_node("danger_scorer", danger_score_node)
    
    # Set flow: news_agent -> crime_agent -> danger_scorer -> END
    workflow.set_entry_point("news_agent")
    workflow.add_edge("news_agent", "crime_agent")
    workflow.add_edge("crime_agent", "danger_scorer")
    workflow.add_edge("danger_scorer", END)
    
    return workflow.compile()

def get_danger_score(latitude: float, longitude: float, street_address: str, 
                    csv_path: str = "crime_data.csv", radius_km: float = 1.0) -> float:
    """
    Main function to get danger score using LangGraph workflow with news and crime data
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate  
        street_address: Street address of location
        csv_path: Path to crime data CSV file
        radius_km: Search radius for crimes in kilometers
        
    Returns:
        float: Danger score from 0.0 (safe) to 1.0 (dangerous)
    """
    # Create workflow
    workflow = create_danger_score_workflow(csv_path, radius_km)
    
    # Initial state
    initial_state = {
        "latitude": latitude,
        "longitude": longitude,
        "street_address": street_address
    }
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state["danger_score"]

# Test function
if __name__ == "__main__":
    # Test the LangGraph workflow
    print("🧠 Testing LangGraph Danger Scorer with News and Crime Agents")
    print("="*60)
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        exit(1)
    
    if not os.getenv("NEWS_API_KEY"):
        print("⚠️  Warning: NEWS_API_KEY not set - news analysis will be skipped")
    
    # Test locations (using SF coordinates to match sample crime data)
    test_locations = [
        (37.7749, -122.4194, "Union Square, San Francisco, CA"),
        (37.7849, -122.4094, "Chinatown, San Francisco, CA"),
        (37.7649, -122.4294, "Mission District, San Francisco, CA")
    ]
    
    for lat, lng, address in test_locations:
        print(f"\n📍 Location: {address}")
        print(f"🔍 Coordinates: ({lat}, {lng})")
        print("📰 Fetching news data...")
        print("🚔 Analyzing crime data...")
        
        # Test with 1km radius
        score = get_danger_score(lat, lng, address, radius_km=1.0, csv_path="/Users/colerobins/Downloads/crime.csv")
        print(f"⚠️  Final Danger Score: {score:.3f}")
        
        print("-" * 40)