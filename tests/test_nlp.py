"""
Unit tests for resq_ai.nlp.command_parser
"""

import pytest
from resq_ai.core.models import VehicleType, OccupantType
from resq_ai.nlp.command_parser import CommandParser


def test_command_parser_acceptance_phrase():
    parser = CommandParser()
    text = "Dispatch rescue boat to Block C. Reports say Main Street is submerged."
    cmd = parser.parse(text)

    assert cmd.intent == "DISPATCH"
    assert cmd.vehicle_type == VehicleType.RESCUE_BOAT
    assert cmd.target_name == "Block C"
    assert cmd.target_location == (22, 34)
    assert cmd.road_to_block == "Main Street"
    assert cmd.hazard_condition == "submerged"


def test_command_parser_block_road_coordinates():
    parser = CommandParser()
    text = "Block road at (15, 20) due to fallen debris"
    cmd = parser.parse(text)

    assert cmd.intent == "BLOCK_ROAD"
    assert cmd.target_location == (15, 20)


def test_command_parser_ambulance_dispatch():
    parser = CommandParser()
    text = "Send ambulance to Central Hospital with 2 elderly patients"
    cmd = parser.parse(text)

    assert cmd.intent == "DISPATCH"
    assert cmd.vehicle_type == VehicleType.AMBULANCE
    assert cmd.occupants_type == OccupantType.ELDERLY
    assert cmd.occupant_count == 2
