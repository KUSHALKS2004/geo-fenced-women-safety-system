import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RISK_GEOJSON = (
    BASE_DIR
    / "gis"
    / "outputs"
    / "safeher_2014_risk_layer.geojson"
)


# ============================================================
# LOAD RISK LAYER
# ============================================================

def load_risk_layer():
    """
    Load the SafeHer-AI predicted 2014 GIS risk layer.
    """

    if not RISK_GEOJSON.exists():
        raise FileNotFoundError(
            f"Risk GeoJSON not found:\n{RISK_GEOJSON}"
        )

    risk_gdf = gpd.read_file(RISK_GEOJSON)

    if risk_gdf.empty:
        raise ValueError("Risk GIS layer is empty.")

    if risk_gdf.crs is None:
        raise ValueError(
            "Risk GIS layer does not have a CRS."
        )

    # Ensure latitude/longitude coordinates
    risk_gdf = risk_gdf.to_crs(epsg=4326)

    return risk_gdf


# ============================================================
# LOCATION CHECK
# ============================================================

def check_location(latitude, longitude):
    """
    Check whether a latitude/longitude falls inside
    one of the mapped SafeHer-AI risk polygons.

    Parameters
    ----------
    latitude : float
        Latitude of the user.

    longitude : float
        Longitude of the user.

    Returns
    -------
    dict
        Geofencing and risk information.
    """

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        raise ValueError(
            "Latitude and longitude must be numeric."
        )

    if not (-90 <= latitude <= 90):
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not (-180 <= longitude <= 180):
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    # --------------------------------------------------------
    # Load GIS risk layer
    # --------------------------------------------------------

    risk_gdf = load_risk_layer()

    # --------------------------------------------------------
    # Create user point
    # --------------------------------------------------------

    user_point = Point(
        longitude,
        latitude
    )

    # --------------------------------------------------------
    # Point-in-polygon
    #
    # covers() includes points lying exactly on a boundary.
    # --------------------------------------------------------

    matches = risk_gdf[
        risk_gdf.geometry.covers(user_point)
    ]

    # --------------------------------------------------------
    # Location outside mapped polygons
    # --------------------------------------------------------

    if matches.empty:

        return {
            "inside_risk_zone": False,
            "latitude": latitude,
            "longitude": longitude,
            "state": None,
            "district": None,
            "predicted_risk_share": None,
            "risk_band": None,
            "message": (
                "Location is outside the currently "
                "mapped SafeHer-AI risk zones."
            )
        }

    # --------------------------------------------------------
    # Get first matching polygon
    # --------------------------------------------------------

    match = matches.iloc[0]

    # --------------------------------------------------------
    # Extract location information
    # --------------------------------------------------------

    state = match.get("STATE_UT")

    district = match.get("DISTRICT")

    predicted_share = match.get(
        "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED"
    )

    risk_band = match.get(
        "PREDICTED_RISK_BAND"
    )

    # --------------------------------------------------------
    # Convert values to standard Python types
    # --------------------------------------------------------

    if state is not None:
        state = str(state)

    if district is not None:
        district = str(district)

    if predicted_share is not None:
        predicted_share = float(predicted_share)

    if risk_band is not None:
        risk_band = str(risk_band)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "inside_risk_zone": True,
        "latitude": latitude,
        "longitude": longitude,
        "state": state,
        "district": district,
        "predicted_risk_share": predicted_share,
        "risk_band": risk_band,
        "message": (
            "Location successfully matched "
            "to a SafeHer-AI risk zone."
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SAFEHER-AI GEOFENCING TEST")
    print("=" * 60)

    latitude = 23.035743284012398
    longitude = 81.39049233657985

    print("\nTesting location:")
    print("Latitude :", latitude)
    print("Longitude:", longitude)

    result = check_location(
        latitude,
        longitude
    )

    print("\nResult:")
    print(result)