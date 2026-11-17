"""
main.py
=======
Command-Line Interface and entrypoint for ResQ-AI.
Supports both Headless CLI simulation runs and Tactical Streamlit UI launch.

Usage:
  python main.py --ui                           # Launches Streamlit dashboard
  python main.py --steps 25                     # Runs headless 25-minute simulation
  python main.py --scenario configs/scenarios/earthquake.yaml --steps 30
  python main.py --command "Dispatch rescue boat to Block C. Reports say Main Street is submerged."
"""

from __future__ import annotations
import argparse
import sys
import os
import subprocess
import time
from typing import Optional

from resq_ai.orchestrator.engine import OrchestratorEngine
from resq_ai.core.models import PriorityLevel


def run_cli_simulation(scenario_path: str, steps: int, command: Optional[str] = None):
    print("\n" + "=" * 70)
    print("      🚨 RESQ-AI: CIVIL DEFENSE EMERGENCY OPERATIONS CENTER 🚨")
    print("=" * 70)
    print(f"Loading Scenario: {scenario_path}")
    print(f"Simulation Horizon: {steps} minutes | Fleet: 5 Ambulances, 3 Boats\n")

    engine = OrchestratorEngine(config_path=scenario_path, seed=42)

    for step_idx in range(1, steps + 1):
        events = engine.step()
        state = engine.get_dashboard_state()
        
        # Periodic summary report every 5 steps
        if step_idx % 5 == 0 or step_idx == 1:
            p1_calls = sum(1 for c in state["unserved_calls"] if c["priority"] == "P1")
            print(
                f"[T+{step_idx:02d}m] Flooded: {state['flooded_count']:3d} cells | "
                f"Blocked: {state['blocked_count']:2d} roads | "
                f"Pending Calls: {len(state['unserved_calls']):2d} (P1: {p1_calls:2d})"
            )

    # Execute custom command if provided
    if command:
        print("\n" + "-" * 70)
        print(f"DISPATCHER COMMAND: \"{command}\"")
        response = engine.handle_natural_language_command(command)
        print(f"RESQ-AI RESPONSE:   {response}")
        print("-" * 70)

    # Final summary report
    final_state = engine.get_dashboard_state()
    print("\n======================= MISSION SUMMARY =======================")
    print(f"Total Simulation Steps:     {final_state['step']} minutes")
    print(f"Total Flooded Urban Cells:  {final_state['flooded_count']} / 2500 ({(final_state['flooded_count']/2500):.1%})")
    print(f"Total Infrastructure Blocks:{final_state['blocked_count']}")
    print(f"Remaining Unserved Calls:   {len(final_state['unserved_calls'])}")
    print(f"D* Local Replan Events:     {engine.replanning_events_count}")
    print("===============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="ResQ-AI Emergency Logistics Assistant")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit Tactical Command Center")
    parser.add_argument("--scenario", type=str, default="configs/scenarios/flash_flood.yaml", help="Path to scenario YAML")
    parser.add_argument("--steps", type=int, default=15, help="Number of simulation steps to run in headless mode")
    parser.add_argument("--command", type=str, default="Dispatch rescue boat to Block C. Reports say Main Street is submerged.",
                        help="Natural language command to execute")

    args = parser.parse_args()

    if args.ui:
        app_path = os.path.join(os.path.dirname(__file__), "resq_ai", "ui", "app.py")
        print(f"Launching Streamlit UI: streamlit run {app_path}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    else:
        run_cli_simulation(scenario_path=args.scenario, steps=args.steps, command=args.command)


if __name__ == "__main__":
    main()
