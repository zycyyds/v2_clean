# Agent 4

Agent 4 owns Step4 task-oriented column selection.

Active code:

- `agent_4/main.py`: standalone Agent4 CLI.
- `agent_4/medical_column_selector/`: task planner, term planner, schema profiler, hard gates, and task-driven column selector.
- `step-4/`: ReAct tools/runtime used by the main orchestrator handoff.

Agent 4 should not contain data quality cleaning workers. Step5 data cleaning belongs to `agent_5`.
