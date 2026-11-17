"""
resq_ai.nlp.command_parser
==========================
Natural Language Dispatch Parser for emergency operations.
Extracts intents and operational entities from dispatcher commands:
- Intent classification (DISPATCH, BLOCK_ROAD, CLEAR_ROAD, REPORT_CALL, QUERY_STATUS)
- Entity extraction: vehicle types, location/blocks, coordinates, street names, flood depths, occupant types
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from resq_ai.core.models import VehicleType, OccupantType, PriorityLevel


@dataclass
class ParsedCommand:
    raw_text: str
    intent: str  # "DISPATCH", "BLOCK_ROAD", "CLEAR_ROAD", "REPORT_CALL", "QUERY_STATUS", "UNKNOWN"
    vehicle_type: Optional[VehicleType] = None
    target_location: Optional[Tuple[int, int]] = None
    target_name: Optional[str] = None  # e.g., "Block C", "Main Street"
    road_to_block: Optional[str] = None
    road_block_location: Optional[Tuple[int, int]] = None
    hazard_condition: Optional[str] = None
    occupants_type: Optional[OccupantType] = None
    occupant_count: int = 1
    confidence: float = 1.0


class CommandParser:
    """Rule-based and regex semantic dispatcher interface."""

    # Pre-mapped named landmarks and arterial streets to grid coordinates
    LANDMARKS: Dict[str, Tuple[int, int]] = {
        "block a": (10, 10),
        "block b": (35, 12),
        "block c": (22, 34),  # Near riverfront / Marina
        "block d": (40, 42),
        "block e": (15, 45),
        "main street": (25, 20),
        "4th avenue": (20, 20),
        "5th street": (25, 30),
        "riverfront": (25, 34),
        "riverfront marina": (25, 34),
        "marina": (25, 34),
        "central hospital": (5, 5),
        "hospital": (5, 5),
        "north uptown": (15, 8),
        "south valley": (30, 45),
        "downtown": (25, 25),
        "bridge 1": (8, 35),
        "bridge 2": (20, 34),
        "bridge 3": (32, 33),
        "bridge 4": (44, 32),
    }

    def parse(self, text: str) -> ParsedCommand:
        text_clean = text.strip()
        lower = text_clean.lower()

        cmd = ParsedCommand(raw_text=text_clean, intent="UNKNOWN")

        # 0. Extract any explicit (x, y) coordinates from text
        coord_matches = re.findall(r"\(?\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)?", lower)
        found_coords = [(int(x), int(y)) for x, y in coord_matches]

        # 1. Check for Road Blockage (e.g. "Reports say Main Street is submerged" or "Block road at (15, 20)")
        block_regex = r"(?:block|collapse|break|close|damage|cut off)\s+(?:road|street|bridge)?\s*(?:at|near)?\s*([a-zA-Z0-9\s,\(\)]+)"
        hazard_regex = r"([a-zA-Z0-9\s]+?)\s+(?:is|has been)?\s*(submerged|flooded|blocked|collapsed|impassable)"
        
        m_block = re.search(block_regex, lower)
        m_hazard = re.search(hazard_regex, lower)

        if m_hazard:
            road_candidate = m_hazard.group(1).strip()
            # Clean leading fillers
            road_candidate = re.sub(r"^(?:reports\s+say\s+|that\s+|the\s+)", "", road_candidate).strip()
            if road_candidate and road_candidate not in ["that", "it", "there"]:
                cmd.road_to_block = road_candidate.title()
                cmd.hazard_condition = m_hazard.group(2)
                if road_candidate in self.LANDMARKS:
                    cmd.road_block_location = self.LANDMARKS[road_candidate]

        # 2. Check for Dispatch intent (e.g. "Dispatch rescue boat to Block C" or "Send ambulance to (10, 10)")
        dispatch_keywords = ["dispatch", "send", "deploy", "assign", "rescue", "route"]
        is_dispatch = any(k in lower for k in dispatch_keywords)

        if is_dispatch:
            cmd.intent = "DISPATCH"
            # Detect vehicle
            if any(w in lower for w in ["boat", "watercraft", "raft"]):
                cmd.vehicle_type = VehicleType.RESCUE_BOAT
            elif any(w in lower for w in ["ambulance", "medic", "paramedic", "van"]):
                cmd.vehicle_type = VehicleType.AMBULANCE
            else:
                # Default based on destination later if unknown
                pass

            # Detect destination name from landmarks
            for lm, coord in self.LANDMARKS.items():
                if lm in lower:
                    # Don't confuse blocked road with destination if both present
                    if cmd.road_to_block and lm.lower() == cmd.road_to_block.lower():
                        continue
                    cmd.target_name = lm.title()
                    cmd.target_location = coord
                    break

            # If coordinate found, assign target location
            if found_coords:
                # If road blockage also has coordinate, differentiate
                if cmd.road_block_location and len(found_coords) > 1:
                    cmd.target_location = found_coords[1]
                elif not cmd.target_location:
                    cmd.target_location = found_coords[0]

        # 3. Pure Road Blockage intent if not dispatch
        elif any(k in lower for k in ["block", "collapse", "close", "break", "submerged", "flooded", "collapsed", "impassable"]):
            cmd.intent = "BLOCK_ROAD"
            if found_coords:
                cmd.target_location = found_coords[0]
                cmd.road_block_location = found_coords[0]
            for lm, coord in self.LANDMARKS.items():
                if lm in lower:
                    cmd.road_to_block = lm.title()
                    cmd.target_location = coord
                    cmd.road_block_location = coord
                    break

        # 4. Road Clear intent
        elif any(k in lower for k in ["clear", "open", "reopened", "repair", "unblock"]):
            cmd.intent = "CLEAR_ROAD"
            if found_coords:
                cmd.target_location = found_coords[0]
            for lm, coord in self.LANDMARKS.items():
                if lm in lower:
                    cmd.road_to_block = lm.title()
                    cmd.target_location = coord
                    break

        # 5. Status Query
        elif any(k in lower for k in ["status", "report", "where", "show", "summary", "how many"]):
            cmd.intent = "QUERY_STATUS"

        # 6. Fallback: if coordinates exist and verb was rescue/help
        if cmd.intent == "UNKNOWN" and found_coords:
            cmd.target_location = found_coords[0]
            if any(w in lower for w in ["help", "victim", "patient", "emergency", "save"]):
                cmd.intent = "DISPATCH"

        # Default vehicle selection if intent is DISPATCH but vehicle was unspecified
        if cmd.intent == "DISPATCH" and cmd.vehicle_type is None:
            if cmd.target_location and cmd.target_location[1] >= 32:
                cmd.vehicle_type = VehicleType.RESCUE_BOAT
            else:
                cmd.vehicle_type = VehicleType.AMBULANCE

        # 7. Extract demographic entities
        if "elderly" in lower or "senior" in lower or "old" in lower:
            cmd.occupants_type = OccupantType.ELDERLY
        elif "child" in lower or "kid" in lower or "baby" in lower:
            cmd.occupants_type = OccupantType.CHILDREN
        elif "adult" in lower:
            cmd.occupants_type = OccupantType.ADULTS

        count_match = re.search(r"(\d+)\s+(?:[a-zA-Z]+\s+)?(?:people|victims|occupants|persons|citizens|patients|residents)", lower)
        if count_match:
            cmd.occupant_count = int(count_match.group(1))

        return cmd
