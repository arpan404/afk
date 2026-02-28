
# Smart Home

A smart home controller agent that uses ToolPolicy on the ToolRegistry for access control, enforcing thermostat range limits and role-based restrictions on locks and cameras.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/35_Smart_Home

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/35_Smart_Home

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/35_Smart_Home

Expected interaction
Your role: guest
User: Unlock the front door
Agent: Policy denied: Only admin users can unlock doors. Current role: guest.
User: Set temperature to 90
Agent: Policy denied: Temperature 90°F is outside safe range (60-85°F).
User: Turn on the bedroom light
Agent: Light 'bedroom_light' is now on (brightness: 80%).

The agent enforces ToolPolicy rules for thermostat range, lock access, and camera privacy.
