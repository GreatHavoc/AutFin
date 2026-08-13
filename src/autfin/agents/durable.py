"""AutFin v7.0 — Durable Execution Configuration

Temporal integration for durable agent runs.
"""
from __future__ import annotations

from typing import Any, Optional

# ponytail: temporalio not installed; lazy imports below
# Only fails if you actually try to use DurableRunner


from autfin.types import AgentDeps
from autfin.agents import supervisor_agent


# ─── Temporal Client ───

_temporal_client = None


async def get_temporal_client(
    address: str = "localhost:7233",
    namespace: str = "default",
):
    """Get or create Temporal client."""
    from temporalio.client import Client
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(
            address,
            namespace=namespace,
            plugins=[PydanticAIPlugin()],
        )
    return _temporal_client


# ─── Durable Agent Capability ───

def create_temporal_durability(
    client=None,
    models: Optional[dict] = None,
):
    """Create TemporalDurability capability for durable execution."""
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    return TemporalDurability(
        client=client,
        models=models,
    )


# ─── Workflow Definition ───

def _make_workflow_class():
    """Create AutFinWorkflow class (needs TemporalDurability on the agent)."""
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow

    class AutFinWorkflow(PydanticAIWorkflow):
        __pydantic_ai_agents__ = [supervisor_agent]

        @property
        def agent(self):
            return supervisor_agent

    return AutFinWorkflow


# ─── Durable Runner ───

class DurableRunner:
    """Run AutFin agents durably via Temporal."""

    def __init__(
        self,
        client=None,
        task_queue: str = "autfin",
        workflow_id_prefix: str = "autfin",
    ):
        self.client = client
        self.task_queue = task_queue
        self.workflow_id_prefix = workflow_id_prefix
        self._worker = None

    async def initialize(self) -> None:
        """Initialize Temporal client and worker."""
        from temporalio.worker import Worker

        if self.client is None:
            self.client = await get_temporal_client()

        # Ensure supervisor has TemporalDurability capability
        from pydantic_ai.durable_exec.temporal import TemporalDurability
        if not any(isinstance(c, TemporalDurability) for c in supervisor_agent._capabilities):
            supervisor_agent._capabilities.append(create_temporal_durability(self.client))

        AutFinWorkflow = _make_workflow_class()
        self._worker = Worker(
            self.client,
            task_queue=self.task_queue,
            workflows=[AutFinWorkflow],
        )
        await self._worker.__aenter__()

    async def close(self) -> None:
        """Close worker and client."""
        if self._worker:
            await self._worker.__aexit__(None, None, None)

    async def run_morning_routine(self, run_id: str, deps: AgentDeps) -> str:
        """Run morning routine durably."""
        workflow_id = f"{self.workflow_id_prefix}-morning-{run_id}"
        AutFinWorkflow = _make_workflow_class()
        result = await self.client.execute_workflow(
            AutFinWorkflow.run,
            args=["Execute morning routine"],
            id=workflow_id,
            task_queue=self.task_queue,
        )
        return result

    async def run_quarterly_review(self, run_id: str, deps: AgentDeps) -> str:
        """Run quarterly review durably."""
        workflow_id = f"{self.workflow_id_prefix}-quarterly-{run_id}"
        AutFinWorkflow = _make_workflow_class()
        result = await self.client.execute_workflow(
            AutFinWorkflow.run,
            args=["Execute quarterly portfolio review"],
            id=workflow_id,
            task_queue=self.task_queue,
        )
        return result

    async def run_price_alert(self, symbol: str, price_move_pct: float, deps: AgentDeps) -> str:
        """Run price alert durably."""
        workflow_id = f"{self.workflow_id_prefix}-alert-{symbol}-{int(price_move_pct*100)}"
        AutFinWorkflow = _make_workflow_class()
        result = await self.client.execute_workflow(
            AutFinWorkflow.run,
            args=[f"Price alert for {symbol}: {price_move_pct:.1f}% move"],
            id=workflow_id,
            task_queue=self.task_queue,
        )
        return result


# ─── Helper Functions ───

async def run_durable_agent(agent, prompt: str, deps: AgentDeps, run_id: str, task_queue: str = "autfin"):
    """Run a single agent durably via Temporal."""
    from pydantic_ai.durable_exec.temporal import TemporalDurability, PydanticAIWorkflow

    client = await get_temporal_client()

    if not any(isinstance(c, TemporalDurability) for c in agent._capabilities):
        agent._capabilities.append(TemporalDurability(client=client))

    workflow_id = f"{agent.name}-{run_id}"

    class SingleAgentWorkflow(PydanticAIWorkflow):
        __pydantic_ai_agents__ = [agent]

    result = await client.execute_workflow(
        SingleAgentWorkflow.run,
        args=[prompt],
        id=workflow_id,
        task_queue=task_queue,
    )
    return result