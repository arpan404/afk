"""
---
name: Smart Home
description: A smart home controller agent that uses ToolPolicy for access control and device-specific restrictions.
tags: [agent, runner, tools, tool-policy, registry, security]
---
---
This example demonstrates how to use a ToolPolicy callback on the ToolRegistry to enforce
access control rules on tool calls. The policy function checks tool name, arguments, and context
to decide whether a call should be allowed or denied. This pattern is essential for building
agents that interact with sensitive systems (IoT devices, databases, APIs) where not every
operation should be unrestricted. The smart home agent controls lights, thermostat, locks,
and cameras, with policy rules that enforce safety limits and role-based access.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolRegistry, ToolContext  # <- @tool, ToolRegistry (with policy support), and ToolContext for runtime info.


# ===========================================================================
# Simulated smart home device state
# ===========================================================================

devices: dict[str, dict] = {  # <- Simulated device state. In a real system, this would be a smart home API (Hue, Nest, etc.).
    "living_room_light": {"type": "light", "on": True, "brightness": 80},
    "bedroom_light": {"type": "light", "on": False, "brightness": 0},
    "kitchen_light": {"type": "light", "on": True, "brightness": 100},
    "thermostat": {"type": "thermostat", "temperature": 72, "mode": "auto"},
    "front_door": {"type": "lock", "locked": True},
    "back_door": {"type": "lock", "locked": True},
    "front_camera": {"type": "camera", "recording": True, "motion_detected": False},
    "backyard_camera": {"type": "camera", "recording": True, "motion_detected": True},
}


# ===========================================================================
# ToolPolicy — the access control callback (the key concept)
# ===========================================================================

def smart_home_policy(tool_name: str, args: dict, ctx: ToolContext) -> None:  # <- A ToolPolicy is a callable: (tool_name, args, ctx) -> None. If the call is allowed, return None. If denied, raise an exception. The ToolRegistry calls this BEFORE every tool invocation.
    """
    Policy rules for the smart home:
    1. Thermostat range: temperature must be between 60-85°F
    2. Lock safety: can only unlock if user_role is 'admin' (not 'guest')
    3. Camera privacy: guests cannot access camera feeds
    """
    user_role = ctx.metadata.get("user_role", "guest") if ctx.metadata else "guest"  # <- Read the user's role from tool context metadata. This is set when calling runner.run().

    # --- Rule 1: Thermostat temperature range ---
    if tool_name == "set_thermostat":
        temp = args.get("temperature", 72)
        if temp < 60 or temp > 85:
            raise ValueError(  # <- Raising any exception from the policy function denies the tool call. The error message is sent back to the agent.
                f"Policy denied: Temperature {temp}°F is outside safe range (60-85°F). "
                f"Please set a temperature between 60 and 85."
            )

    # --- Rule 2: Lock access control ---
    if tool_name in ("lock_door", "unlock_door"):
        if "unlock" in tool_name and user_role != "admin":
            raise PermissionError(  # <- Use PermissionError for access control denials.
                f"Policy denied: Only admin users can unlock doors. "
                f"Current role: {user_role}. Contact a household admin."
            )

    # --- Rule 3: Camera privacy ---
    if tool_name == "view_camera":
        if user_role == "guest":
            raise PermissionError(
                f"Policy denied: Guests cannot access camera feeds. "
                f"Current role: {user_role}."
            )


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class ToggleLightArgs(BaseModel):
    device_name: str = Field(description="Name of the light device (e.g., 'living_room_light')")
    on: bool = Field(description="True to turn on, False to turn off")
    brightness: int = Field(default=80, description="Brightness level 0-100 (only when turning on)")


class ThermostatArgs(BaseModel):
    temperature: int = Field(description="Target temperature in Fahrenheit")
    mode: str = Field(default="auto", description="Mode: auto, cool, heat, off")


class DoorArgs(BaseModel):
    device_name: str = Field(description="Name of the door lock (e.g., 'front_door')")


class CameraArgs(BaseModel):
    device_name: str = Field(description="Name of the camera (e.g., 'front_camera')")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=ToggleLightArgs, name="toggle_light", description="Turn a light on or off and set brightness")
def toggle_light(args: ToggleLightArgs) -> str:
    device = devices.get(args.device_name)
    if device is None or device["type"] != "light":
        available = [k for k, v in devices.items() if v["type"] == "light"]
        return f"Light '{args.device_name}' not found. Available lights: {', '.join(available)}"

    device["on"] = args.on
    device["brightness"] = args.brightness if args.on else 0
    state = f"on (brightness: {device['brightness']}%)" if device["on"] else "off"
    return f"Light '{args.device_name}' is now {state}."


@tool(args_model=ThermostatArgs, name="set_thermostat", description="Set the thermostat temperature and mode")
def set_thermostat(args: ThermostatArgs) -> str:  # <- Note: the policy checks temperature BEFORE this tool runs. If the policy raises, this function never executes.
    devices["thermostat"]["temperature"] = args.temperature
    devices["thermostat"]["mode"] = args.mode
    return f"Thermostat set to {args.temperature}°F, mode: {args.mode}."


@tool(args_model=DoorArgs, name="unlock_door", description="Unlock a door — requires admin role")
def unlock_door(args: DoorArgs) -> str:  # <- Policy checks user_role before this executes.
    device = devices.get(args.device_name)
    if device is None or device["type"] != "lock":
        return f"Lock '{args.device_name}' not found."
    device["locked"] = False
    return f"Door '{args.device_name}' UNLOCKED."


@tool(args_model=DoorArgs, name="lock_door", description="Lock a door")
def lock_door(args: DoorArgs) -> str:
    device = devices.get(args.device_name)
    if device is None or device["type"] != "lock":
        return f"Lock '{args.device_name}' not found."
    device["locked"] = True
    return f"Door '{args.device_name}' locked."


@tool(args_model=CameraArgs, name="view_camera", description="View a camera feed — restricted to admin users")
def view_camera(args: CameraArgs) -> str:  # <- Policy checks user_role before this executes.
    device = devices.get(args.device_name)
    if device is None or device["type"] != "camera":
        return f"Camera '{args.device_name}' not found."
    motion = "Motion detected!" if device["motion_detected"] else "No motion"
    recording = "Recording" if device["recording"] else "Not recording"
    return f"Camera '{args.device_name}': {recording} | {motion}"


@tool(args_model=EmptyArgs, name="get_home_status", description="Get the status of all smart home devices")
def get_home_status(args: EmptyArgs) -> str:
    lines = ["Smart Home Status:"]
    for name, device in devices.items():
        if device["type"] == "light":
            state = f"on ({device['brightness']}%)" if device["on"] else "off"
            lines.append(f"  {name}: {state}")
        elif device["type"] == "thermostat":
            lines.append(f"  {name}: {device['temperature']}°F ({device['mode']})")
        elif device["type"] == "lock":
            lines.append(f"  {name}: {'locked' if device['locked'] else 'UNLOCKED'}")
        elif device["type"] == "camera":
            motion = "motion!" if device["motion_detected"] else "clear"
            lines.append(f"  {name}: {'recording' if device['recording'] else 'off'} ({motion})")
    return "\n".join(lines)


# ===========================================================================
# ToolRegistry with policy
# ===========================================================================

registry = ToolRegistry(  # <- Create a ToolRegistry with the policy callback attached.
    policy=smart_home_policy,  # <- The policy function is called before every tool invocation. If it raises, the tool call is denied and the error is sent to the agent.
)

registry.register(toggle_light)
registry.register(set_thermostat)
registry.register(unlock_door)
registry.register(lock_door)
registry.register(view_camera)
registry.register(get_home_status)


# ===========================================================================
# Agent and runner setup
# ===========================================================================

home_agent = Agent(
    name="smart-home",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a smart home assistant. You control lights, thermostat, doors, and cameras.

    Available commands:
    - Lights: turn on/off, adjust brightness
    - Thermostat: set temperature (60-85°F) and mode
    - Doors: lock/unlock (unlock needs admin role)
    - Cameras: view feeds (needs admin role)
    - Status: get overview of all devices

    If a command is denied by the policy, explain the restriction to the user.
    Suggest alternatives when possible (e.g., "You can lock the door as a guest, but unlocking requires admin access").

    **NOTE**: The system enforces safety and access policies automatically!
    """,
    tools=registry.list(),
)

runner = Runner()

if __name__ == "__main__":
    print("Smart Home Controller (type 'quit' to exit)")
    print("=" * 50)

    # --- Let the user choose their role ---
    role = input("Your role (admin/guest) [guest]: ").strip().lower() or "guest"

    print(f"\nLogged in as: {role}")
    print("Try: 'turn off the bedroom light', 'set temperature to 75', 'unlock the front door'\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = runner.run_sync(
            home_agent,
            user_message=user_input,
            context={"user_role": role},  # <- Pass the user's role as context. Tools and policy access this via ToolContext.metadata.
        )

        print(f"[smart-home] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a smart home controller with a ToolPolicy callback on the ToolRegistry that
enforces access control: thermostat range limits (60-85°F), lock access restricted to admin role, and
camera feeds restricted to admin role. The policy function receives tool_name, args, and ToolContext,
and raises exceptions to deny calls. This pattern provides deterministic, code-level security for
tool invocations independent of LLM behavior.
---
---
What's next?
- Try switching between admin and guest roles to see different access levels.
- Add time-based policies (e.g., no thermostat changes after 11 PM).
- Combine ToolPolicy with PolicyEngine rules for layered security.
- Add a "policy_log" that records all allowed and denied actions for audit.
- Implement a "request_access" tool where guests can ask admins for temporary permissions.
- Check out the Content Moderator example for PolicyEngine-based approval workflows!
---
"""
