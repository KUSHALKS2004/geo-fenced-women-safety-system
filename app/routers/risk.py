from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gis.geofencing.risk_geofence import check_location


router = APIRouter(
    prefix="/risk",
    tags=["Risk Assessment"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class LocationRequest(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="User latitude"
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="User longitude"
    )


# ============================================================
# RISK CHECK ENDPOINT
# ============================================================

@router.post("/check")
def check_risk(location: LocationRequest):

    try:

        result = check_location(
            location.latitude,
            location.longitude
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {error}"
        )