"""
---
name: Travel Planner
description: A travel planner that uses DelegationPlan with quorum and allow_optional_failures join policies for resilient multi-agent orchestration.
tags: [agent, runner, delegation, delegation-plan, join-policy, quorum, async]
---
---
This example demonstrates DelegationPlan's JoinPolicy options for handling partial success in
multi-agent workflows. Three specialist subagents (flights, hotels, activities) search for travel
options in parallel. With join_policy="allow_optional_failures", the orchestrator succeeds even
if one agent fails -- it uses results from whichever agents succeed. The example also shows the
"quorum" option (e.g., quorum=2: need at least 2 of 3 agents to succeed). This pattern is ideal
for scenarios where partial results are better than no results, and individual agent failures
should not block the entire workflow.
---
"""

import asyncio  # <- Async required for delegation engine and parallel agent execution.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution and delegation.
from afk.agents import Agent  # <- Agent defines each specialist subagent.
from afk.agents.delegation import (  # <- Delegation system for parallel orchestration with join policies.
    DelegationPlan,  # <- The plan: nodes (agents), edges (dependencies), and join policy.
    DelegationNode,  # <- One agent invocation in the delegation plan.
    DelegationEdge,  # <- Dependency edge between nodes (not used here since all agents run in parallel).
    RetryPolicy,  # <- Per-node retry configuration.
)
from afk.tools import tool  # <- @tool decorator for creating agent-callable tools.


# ===========================================================================
# Simulated travel data — three categories of travel information
# ===========================================================================

FLIGHT_DATA: dict[str, list[dict]] = {  # <- Simulated flight search results keyed by destination city.
    "tokyo": [
        {"airline": "ANA", "flight": "NH005", "departure": "10:30 AM", "arrival": "2:45 PM +1", "price": "$1,150", "class": "Economy", "stops": "Direct"},
        {"airline": "JAL", "flight": "JL062", "departure": "1:15 PM", "arrival": "4:30 PM +1", "price": "$1,280", "class": "Economy", "stops": "Direct"},
        {"airline": "United", "flight": "UA837", "departure": "11:00 AM", "arrival": "5:20 PM +1", "price": "$980", "class": "Economy", "stops": "1 stop (SFO)"},
    ],
    "paris": [
        {"airline": "Air France", "flight": "AF001", "departure": "6:30 PM", "arrival": "8:45 AM +1", "price": "$890", "class": "Economy", "stops": "Direct"},
        {"airline": "Delta", "flight": "DL264", "departure": "10:00 PM", "arrival": "11:30 AM +1", "price": "$820", "class": "Economy", "stops": "Direct"},
    ],
    "default": [
        {"airline": "International Air", "flight": "IA100", "departure": "9:00 AM", "arrival": "6:00 PM", "price": "$750", "class": "Economy", "stops": "1 stop"},
    ],
}

HOTEL_DATA: dict[str, list[dict]] = {  # <- Simulated hotel search results keyed by destination city.
    "tokyo": [
        {"name": "Park Hyatt Tokyo", "rating": "5-star", "price": "$450/night", "area": "Shinjuku", "highlights": "Iconic skyline views, pool, spa"},
        {"name": "Hotel Gracery Shinjuku", "rating": "4-star", "price": "$180/night", "area": "Kabukicho", "highlights": "Godzilla terrace, central location"},
        {"name": "Sakura Hotel Jimbocho", "rating": "3-star", "price": "$85/night", "area": "Chiyoda", "highlights": "Budget-friendly, 24h cafe, cultural area"},
    ],
    "paris": [
        {"name": "Le Meurice", "rating": "5-star", "price": "$650/night", "area": "1st Arr.", "highlights": "Tuileries views, Michelin dining"},
        {"name": "Hotel Fabric", "rating": "3-star", "price": "$140/night", "area": "11th Arr.", "highlights": "Boutique, Oberkampf nightlife"},
    ],
    "default": [
        {"name": "City Center Hotel", "rating": "3-star", "price": "$120/night", "area": "Downtown", "highlights": "Central, business-friendly"},
    ],
}

ACTIVITY_DATA: dict[str, list[dict]] = {  # <- Simulated local activity/attraction data keyed by destination city.
    "tokyo": [
        {"name": "Tsukiji Outer Market Food Tour", "duration": "3 hours", "price": "$65", "category": "Food & Culture", "rating": "4.8/5"},
        {"name": "Meiji Shrine & Harajuku Walk", "duration": "2 hours", "price": "Free", "category": "Culture & History", "rating": "4.7/5"},
        {"name": "TeamLab Borderless", "duration": "2-3 hours", "price": "$30", "category": "Art & Technology", "rating": "4.9/5"},
        {"name": "Akihabara Electric Town Tour", "duration": "3 hours", "price": "$45", "category": "Shopping & Tech", "rating": "4.5/5"},
        {"name": "Mt. Fuji Day Trip", "duration": "Full day", "price": "$120", "category": "Nature & Adventure", "rating": "4.6/5"},
    ],
    "paris": [
        {"name": "Louvre Museum Skip-the-line", "duration": "3 hours", "price": "$55", "category": "Art & History", "rating": "4.7/5"},
        {"name": "Seine River Cruise", "duration": "1 hour", "price": "$20", "category": "Sightseeing", "rating": "4.5/5"},
        {"name": "Montmartre Walking Tour", "duration": "2 hours", "price": "$35", "category": "Culture & History", "rating": "4.6/5"},
    ],
    "default": [
        {"name": "City Walking Tour", "duration": "2 hours", "price": "$25", "category": "Sightseeing", "rating": "4.3/5"},
    ],
}


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class SearchArgs(BaseModel):  # <- Shared schema for all three search tools. Each takes a destination and optional number of days.
    destination: str = Field(description="The travel destination city name")
    days: int = Field(description="Number of days for the trip", default=5)


# ===========================================================================
# Specialist tools — one per subagent
# ===========================================================================

@tool(  # <- Flight search tool. Returns simulated flight options for the given destination.
    args_model=SearchArgs,
    name="search_flights",
    description="Search for available flights to a destination. Returns flight options with airlines, times, prices, and stop information.",
)
def search_flights(args: SearchArgs) -> str:
    city = args.destination.lower()
    flights = FLIGHT_DATA.get(city, FLIGHT_DATA["default"])  # <- Fall back to default data if the city isn't in our database.

    lines = [f"Flight Options to {args.destination.title()}:", "=" * 45]
    for f in flights:
        lines.append(
            f"  {f['airline']} {f['flight']}: {f['departure']} -> {f['arrival']}\n"
            f"    Price: {f['price']} ({f['class']}) | {f['stops']}"
        )
    lines.append(f"\n  Showing {len(flights)} flights for round-trip ({args.days} days)")
    return "\n".join(lines)


@tool(  # <- Hotel search tool. Returns simulated hotel options for the given destination.
    args_model=SearchArgs,
    name="search_hotels",
    description="Search for available hotels at a destination. Returns hotel options with ratings, prices, areas, and highlights.",
)
def search_hotels(args: SearchArgs) -> str:
    city = args.destination.lower()
    hotels = HOTEL_DATA.get(city, HOTEL_DATA["default"])

    lines = [f"Hotel Options in {args.destination.title()}:", "=" * 45]
    for h in hotels:
        lines.append(
            f"  {h['name']} ({h['rating']})\n"
            f"    {h['price']} | Area: {h['area']}\n"
            f"    Highlights: {h['highlights']}"
        )
    total_cost_range = f"${int(hotels[-1]['price'].replace('$', '').replace('/night', '')) * args.days} - ${int(hotels[0]['price'].replace('$', '').replace('/night', '')) * args.days}"
    lines.append(f"\n  {args.days}-night stay cost range: {total_cost_range}")
    return "\n".join(lines)


@tool(  # <- Activity search tool. Returns simulated local activities and attractions.
    args_model=SearchArgs,
    name="search_activities",
    description="Search for local activities and attractions at a destination. Returns tours, experiences, and sightseeing options.",
)
def search_activities(args: SearchArgs) -> str:
    city = args.destination.lower()
    activities = ACTIVITY_DATA.get(city, ACTIVITY_DATA["default"])

    lines = [f"Activities in {args.destination.title()}:", "=" * 45]
    for a in activities:
        lines.append(
            f"  {a['name']}\n"
            f"    Duration: {a['duration']} | Price: {a['price']}\n"
            f"    Category: {a['category']} | Rating: {a['rating']}"
        )
    total_price = sum(int(a["price"].replace("$", "").replace("Free", "0")) for a in activities)
    lines.append(f"\n  Total activities cost (all): ${total_price}")
    lines.append(f"  Recommended for a {args.days}-day trip: pick {min(args.days, len(activities))} activities")
    return "\n".join(lines)


# ===========================================================================
# Step 1: Define specialist subagents — one per travel category
# ===========================================================================

flights_agent = Agent(  # <- Flight search specialist.
    name="flights-agent",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a flight search specialist. Use the search_flights tool to find flight options
    for the given destination. Present the results clearly with prices, times, and recommendations.
    Recommend the best value option.
    """,
    tools=[search_flights],  # <- Each specialist has exactly one tool.
)

hotels_agent = Agent(  # <- Hotel search specialist.
    name="hotels-agent",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a hotel search specialist. Use the search_hotels tool to find accommodation
    options. Present options across different budgets (luxury, mid-range, budget) and
    recommend based on value and location.
    """,
    tools=[search_hotels],
)

activities_agent = Agent(  # <- Activities and attractions specialist.
    name="activities-agent",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a local activities expert. Use the search_activities tool to find things to do
    at the destination. Organize activities by category and suggest a day-by-day itinerary
    based on the trip duration.
    """,
    tools=[search_activities],
)


# ===========================================================================
# Step 2: Create DelegationPlans with different JoinPolicy options
# ===========================================================================
# DelegationPlan orchestrates how subagents run. The key difference from simple
# subagents is the join_policy, which controls what happens when some agents
# succeed and others fail.

# --- Plan A: allow_optional_failures ---
# Use results from whatever agents succeed. If one fails, the others still contribute.
# This is the most resilient option -- partial results are better than no results.

plan_optional_failures = DelegationPlan(  # <- This plan runs all three agents in parallel with no dependencies between them (no edges).
    nodes=[
        DelegationNode(
            node_id="flights",
            target_agent="flights-agent",  # <- Must match the agent's name field.
            input_binding={"task": "Search for flights"},  # <- Context passed to the agent.
            timeout_s=30.0,
            required=False,  # <- Mark as NOT required. If this node fails, the plan still succeeds. This is key for allow_optional_failures.
        ),
        DelegationNode(
            node_id="hotels",
            target_agent="hotels-agent",
            input_binding={"task": "Search for hotels"},
            timeout_s=30.0,
            required=False,  # <- Also optional. The plan tolerates any individual failure.
        ),
        DelegationNode(
            node_id="activities",
            target_agent="activities-agent",
            input_binding={"task": "Search for activities"},
            timeout_s=30.0,
            required=False,  # <- All three are optional, so even if two fail, the plan returns the one that succeeded.
        ),
    ],
    edges=[],  # <- No edges = no dependencies. All three agents run in parallel.
    join_policy="allow_optional_failures",  # <- The plan succeeds as long as at least one non-required node succeeds. Failed nodes are reported but don't block the result.
    max_parallelism=3,  # <- All three agents run concurrently. Set to 2 to limit to 2 at a time.
)

# --- Plan B: quorum ---
# Require at least N of M agents to succeed. Useful when you want redundancy
# but don't need 100% completion.

plan_quorum = DelegationPlan(  # <- Same nodes and edges as Plan A, but with quorum join policy.
    nodes=[
        DelegationNode(
            node_id="flights",
            target_agent="flights-agent",
            input_binding={"task": "Search for flights"},
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2, backoff_base_s=1.0),  # <- Retry once before marking as failed. Gives transient failures a second chance.
        ),
        DelegationNode(
            node_id="hotels",
            target_agent="hotels-agent",
            input_binding={"task": "Search for hotels"},
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2, backoff_base_s=1.0),
        ),
        DelegationNode(
            node_id="activities",
            target_agent="activities-agent",
            input_binding={"task": "Search for activities"},
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2, backoff_base_s=1.0),
        ),
    ],
    edges=[],  # <- No dependencies -- all run in parallel.
    join_policy="quorum",  # <- The plan succeeds when at least `quorum` nodes complete successfully.
    quorum=2,  # <- Need at least 2 of 3 agents to succeed. If only 1 succeeds, the plan is marked as failed. If 2 or 3 succeed, it's a success.
    max_parallelism=3,  # <- All three run concurrently.
)

"""
DelegationPlan visualization — both plans have the same topology:

    [flights]    [hotels]    [activities]
         \\          |          /
          \\         |         /
           \\        |        /
            [orchestrator joins results]

- All three agents run in parallel (no edges between them)
- Plan A (allow_optional_failures): any subset of successes is accepted
- Plan B (quorum=2): at least 2 must succeed for the plan to pass
"""


# ===========================================================================
# Step 3: Create the orchestrator agent
# ===========================================================================

orchestrator = Agent(
    name="travel-planner",  # <- The top-level agent the user interacts with.
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a travel planning assistant. When a user asks to plan a trip, delegate to your
    three specialist subagents to search for flights, hotels, and activities in parallel.

    After receiving results from your subagents, combine them into a comprehensive travel plan:
    1. Flight recommendations (best value and most convenient)
    2. Hotel options across different budgets
    3. Activity itinerary organized by day
    4. Estimated total trip cost

    If some subagents fail (e.g., no flights found), work with whatever results you received
    and note what information is missing. Partial plans are better than no plan at all.
    """,  # <- Instructions tell the agent to handle partial failures gracefully, matching the allow_optional_failures policy.
    subagents=[flights_agent, hotels_agent, activities_agent],  # <- Register all three specialists as subagents. The Runner routes delegated work to these.
)

runner = Runner()


# ===========================================================================
# Step 4: Run the travel planner
# ===========================================================================

async def main():
    print("Travel Planner Agent")
    print("=" * 50)
    print("Plan trips with parallel flight, hotel, and activity searches.")
    print("Three specialist agents work simultaneously to build your itinerary.")
    print()

    # --- Show which join policy is active ---
    print("DelegationPlan join policies demonstrated:")
    print("  Plan A: allow_optional_failures — succeeds even if some agents fail")
    print("  Plan B: quorum=2 — needs at least 2 of 3 agents to succeed")
    print()

    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("[] > ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Happy travels!")
            break

        if not user_input:
            user_input = "Plan a 5-day trip to Tokyo."  # <- Default input that exercises all three specialist agents.

        print(f"\nSearching flights, hotels, and activities in parallel...\n")

        # --- Run with the orchestrator agent ---
        response = await runner.run(  # <- The Runner executes the orchestrator, which delegates to subagents. The delegation plan controls parallel execution and failure handling.
            orchestrator,
            user_message=user_input,
        )

        print(f"[travel-planner] > {response.final_text}")

        # --- Show delegation details if available ---
        if hasattr(response, "subagent_calls") and response.subagent_calls:
            print(f"\n--- Delegation Summary ---")
            print(f"  Subagent calls: {len(response.subagent_calls)}")
            for sub in response.subagent_calls:
                status = "ok" if sub.success else "failed"
                print(f"    {sub.target_agent}: [{status}]")

        print()  # <- Blank line between rounds.


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop for our async main function.



"""
---
Tl;dr: This example creates a travel planner with three specialist subagents (flights, hotels,
activities) orchestrated via DelegationPlan with different JoinPolicy options. Plan A uses
join_policy="allow_optional_failures" with required=False nodes, so the plan succeeds even if
some agents fail -- it uses whatever results are available. Plan B uses join_policy="quorum"
with quorum=2, requiring at least 2 of 3 agents to succeed. Both plans run all three agents in
parallel (max_parallelism=3) with no dependencies between them. The orchestrator agent combines
results from successful agents into a comprehensive travel plan. This pattern is ideal for
scenarios where partial results are acceptable and individual failures should not block the
entire workflow.
---
---
What's next?
- Try adding a fourth specialist (e.g., "visa-agent" for visa requirements) and see how it slots into the parallel plan.
- Experiment with join_policy="first_success" to get the fastest result and ignore slower agents — useful for redundant searches.
- Add DelegationEdge dependencies to create a two-phase plan: first search flights, then search hotels near the airport.
- Set required=True on the flights node to make it mandatory while keeping hotels and activities optional.
- Inject a failure (make a tool raise an exception) to see how allow_optional_failures and quorum handle it differently.
- Combine delegation with memory (see the Flashcard Tutor example) to remember user travel preferences across sessions!
---
"""
