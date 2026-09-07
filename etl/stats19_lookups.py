"""STATS19 numeric code -> human-readable label dictionaries.

Source: DfT "Road Safety Open Dataset Data Guide" variable lookups.
Every dictionary maps an integer code to a label; -1 always means
"Data missing or out of range" and is preserved as its own labeled
category rather than dropped (STATS19's own convention).

Codes not present in a dictionary decode to "Unknown code {n}" instead of
raising, so an unexpected/rare code never crashes the pipeline -- it just
shows up as its own visible category for later review.
"""

from __future__ import annotations

import pandas as pd

SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight", -1: "Data missing"}

DAY_OF_WEEK = {
    1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
    5: "Thursday", 6: "Friday", 7: "Saturday",
}

URBAN_OR_RURAL_AREA = {1: "Urban", 2: "Rural", 3: "Unallocated", -1: "Data missing"}

DID_POLICE_OFFICER_ATTEND = {
    1: "Yes",
    2: "No - self-completion form",
    3: "No - accident reported by other means",
    -1: "Data missing",
}

TRUNK_ROAD_FLAG = {1: "Trunk road", 2: "Non-trunk road", -1: "Data missing"}

ROAD_TYPE = {
    1: "Roundabout", 2: "One way street", 3: "Dual carriageway",
    6: "Single carriageway", 7: "Slip road", 9: "Unknown",
    12: "One way street/Slip road", -1: "Data missing",
}

JUNCTION_DETAIL = {
    0: "Not at junction or within 20 metres",
    1: "Roundabout", 2: "Mini-roundabout", 3: "T or staggered junction",
    5: "Slip road", 6: "Crossroads", 7: "More than 4 arms (not roundabout)",
    8: "Private drive or entrance", 9: "Other junction", -1: "Data missing",
}

JUNCTION_CONTROL = {
    0: "Not at junction or within 20 metres",
    1: "Authorised person", 2: "Auto traffic signal", 3: "Stop sign",
    4: "Give way or uncontrolled", -1: "Data missing / Not at junction",
}

PEDESTRIAN_CROSSING = {
    0: "No physical crossing facility within 50m",
    1: "Zebra crossing",
    4: "Pelican, puffin, toucan or similar non-junction light crossing",
    5: "Pedestrian phase at traffic signal junction",
    7: "Footbridge or subway",
    8: "Central refuge",
    -1: "Data missing",
}

LIGHT_CONDITIONS = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Data missing",
}

WEATHER_CONDITIONS = {
    1: "Fine, no high winds", 2: "Raining, no high winds",
    3: "Snowing, no high winds", 4: "Fine + high winds",
    5: "Raining + high winds", 6: "Snowing + high winds",
    7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Data missing",
}

ROAD_SURFACE_CONDITIONS = {
    1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice",
    5: "Flood over 3cm deep", 6: "Oil or diesel", 7: "Mud",
    -1: "Data missing",
}

SPECIAL_CONDITIONS_AT_SITE = {
    0: "None", 1: "Auto traffic signal - out", 2: "Auto signal partially defective",
    3: "Road sign or marking defective or obscured", 4: "Roadworks",
    5: "Road surface defective", 6: "Oil or diesel", 7: "Mud", -1: "Data missing",
}

CARRIAGEWAY_HAZARDS = {
    0: "None", 1: "Vehicle load on road", 2: "Other object on road",
    3: "Previous accident", 4: "Dog on road", 5: "Other animal on road",
    6: "Pedestrian in carriageway (not injured)",
    7: "Any animal in carriageway (except ridden horse)", -1: "Data missing",
}

ROAD_CLASS = {
    1: "Motorway", 2: "A(M)", 3: "A", 4: "B", 5: "C",
    6: "Unclassified", 0: "Not at junction or within 20 metres",
    -1: "Data missing",
}

CASUALTY_CLASS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}

SEX = {1: "Male", 2: "Female", 3: "Not known", 9: "Unknown", -1: "Data missing"}

AGE_BAND = {
    1: "0-5", 2: "6-10", 3: "11-15", 4: "16-20", 5: "21-25", 6: "26-35",
    7: "36-45", 8: "46-55", 9: "56-65", 10: "66-75", 11: "Over 75",
    -1: "Data missing",
}

PEDESTRIAN_LOCATION = {
    0: "Not a pedestrian",
    1: "Crossing on pedestrian crossing facility",
    2: "Crossing in zig-zag approach lines",
    3: "Crossing in zig-zag exit lines",
    4: "Crossing elsewhere within 50m of pedestrian crossing",
    5: "In carriageway, crossing - not at crossing facility",
    6: "On footway or verge",
    7: "On refuge, central island or central reservation",
    8: "In centre of carriageway, not on refuge/island/reservation",
    9: "In carriageway, not crossing, walking along, facing traffic",
    10: "In carriageway, not crossing, walking along, back to traffic",
    -1: "Data missing / unknown or other",
}

PEDESTRIAN_MOVEMENT = {
    0: "Not a pedestrian",
    1: "Crossing from driver's nearside",
    2: "Crossing from nearside, masked by parked/stationary vehicle",
    3: "Crossing from driver's offside",
    4: "Crossing from offside, masked by parked/stationary vehicle",
    5: "In carriageway, stationary - not crossing",
    6: "In carriageway, stationary - not crossing, masked by vehicle",
    7: "Walking along in carriageway, facing traffic",
    8: "Walking along in carriageway, back to traffic",
    9: "Unknown or other",
    -1: "Data missing",
}

CAR_PASSENGER = {
    0: "Not car passenger", 1: "Front seat passenger",
    2: "Rear seat passenger", 9: "Unknown", -1: "Data missing",
}

BUS_OR_COACH_PASSENGER = {
    0: "Not a bus or coach passenger", 1: "Boarding", 2: "Alighting",
    3: "Standing passenger", 4: "Seated passenger", -1: "Data missing",
}

CASUALTY_TYPE = {
    0: "Pedestrian", 1: "Cyclist",
    2: "Motorcycle 50cc and under rider or passenger",
    3: "Motorcycle 125cc and under rider or passenger",
    4: "Motorcycle over 125cc and up to 500cc rider or passenger",
    5: "Motorcycle over 500cc rider or passenger",
    8: "Taxi/Private hire car occupant", 9: "Car occupant",
    10: "Minibus occupant", 11: "Bus or coach occupant",
    16: "Horse rider", 17: "Agricultural vehicle occupant",
    18: "Tram occupant", 19: "Van/Goods 3.5T mgw or under occupant",
    20: "Goods over 3.5t and under 7.5t occupant",
    21: "Goods 7.5 tonnes mgw and over occupant",
    22: "Mobility scooter rider", 23: "Electric motorcycle rider or passenger",
    90: "Other vehicle occupant",
    97: "Motorcycle - unknown cc rider or passenger",
    98: "Goods vehicle - unknown weight occupant",
    99: "Unknown vehicle type occupant", -1: "Data missing",
}

VEHICLE_TYPE = {
    1: "Pedal cycle", 2: "Motorcycle 50cc and under",
    3: "Motorcycle 125cc and under",
    4: "Motorcycle over 125cc and up to 500cc",
    5: "Motorcycle over 500cc", 8: "Taxi/Private hire car", 9: "Car",
    10: "Minibus", 11: "Bus or coach", 16: "Ridden horse",
    17: "Agricultural vehicle", 18: "Tram",
    19: "Van/Goods 3.5T mgw or under",
    20: "Goods over 3.5t and under 7.5t",
    21: "Goods 7.5 tonnes mgw and over", 22: "Mobility scooter",
    23: "Electric motorcycle", 90: "Other vehicle",
    97: "Motorcycle - unknown cc", 98: "Goods vehicle - unknown weight",
    99: "Unknown vehicle type", -1: "Data missing",
}

TOWING_AND_ARTICULATION = {
    0: "No tow/articulation", 1: "Articulated vehicle",
    2: "Double or multiple trailer", 3: "Caravan", 4: "Single trailer",
    5: "Other tow", -1: "Data missing",
}

VEHICLE_MANOEUVRE = {
    1: "Reversing", 2: "Parked", 3: "Waiting to go - held up",
    4: "Slowing or stopping", 5: "Moving off", 6: "U-turn",
    7: "Turning left", 8: "Waiting to turn left", 9: "Turning right",
    10: "Waiting to turn right", 11: "Changing lane to left",
    12: "Changing lane to right", 13: "Overtaking moving vehicle - offside",
    14: "Overtaking static vehicle - offside", 15: "Overtaking - nearside",
    16: "Going ahead left-hand bend", 17: "Going ahead right-hand bend",
    18: "Going ahead other", -1: "Data missing",
}

JUNCTION_LOCATION = {
    0: "Not at or within 20 metres of junction",
    1: "Approaching junction or waiting/parked at junction approach",
    2: "Cleared junction or waiting/parked at junction exit",
    3: "Leaving roundabout", 4: "Entering roundabout",
    5: "Leaving main road", 6: "Entering main road",
    7: "Entering from slip road",
    8: "Mid junction - on roundabout or main road", -1: "Data missing",
}

SKIDDING_AND_OVERTURNING = {
    0: "None", 1: "Skidded", 2: "Skidded and overturned",
    3: "Jack-knifed", 4: "Jack-knifed and overturned", 5: "Overturned",
    -1: "Data missing",
}

FIRST_POINT_OF_IMPACT = {
    0: "Did not impact", 1: "Front", 2: "Back", 3: "Offside",
    4: "Nearside", -1: "Data missing",
}

PROPULSION_CODE = {
    1: "Petrol", 2: "Heavy oil (diesel)", 3: "Electric", 4: "Steam",
    5: "Gas", 6: "Petrol/Gas (LPG)", 7: "Gas/Bi-fuel", 8: "Hybrid electric",
    9: "Gas Diesel", 10: "New fuel technology", 11: "Fuel cells",
    12: "Electric diesel", -1: "Data missing",
}

JOURNEY_PURPOSE = {
    1: "Journey as part of work", 2: "Commuting to/from work",
    3: "Taking pupil to/from school", 4: "Pupil riding to/from school",
    5: "Other", 6: "Not known", -1: "Data missing",
}

VEHICLE_LEFT_HAND_DRIVE = {1: "No", 2: "Yes", 9: "Not known", -1: "Data missing"}


def decode(lookup: dict, code) -> str:
    """Decode a single STATS19 code; unmapped/NaN codes never raise."""
    if pd.isna(code):
        return "Data missing"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return f"Unknown code {code!r}"
    return lookup.get(code, f"Unknown code {code}")


def decode_series(series: pd.Series, lookup: dict) -> pd.Series:
    """Vectorized decode of a whole column against a lookup dict."""
    numeric = pd.to_numeric(series, errors="coerce")
    labels = numeric.map(lookup)
    missing_mask = numeric.isna()
    unknown_mask = labels.isna() & ~missing_mask
    labels = labels.where(~missing_mask, "Data missing")
    if unknown_mask.any():
        labels = labels.where(
            ~unknown_mask,
            "Unknown code " + numeric.astype("Int64").astype(str),
        )
    return labels
