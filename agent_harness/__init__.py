"""agent_harness: one task in, the right model per sub-task out, with the receipts.

The design is a while loop over a blackboard, on purpose. Every decision point
(triage, plan, routing) is a strict JSON contract that is validated before it is
trusted, and every step lands in an append-only trace you can read afterwards.
"""

__version__ = "0.3.0"
SERVICE = "agent-harness"
