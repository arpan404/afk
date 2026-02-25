"""Custom execution contract for queue-based SLA escalations."""

from afk.queues import ExecutionContractContext, TaskItem


class SLAEscalationContract:
    contract_id = "sla.escalation.v1"
    requires_agent = False

    async def execute(
        self,
        task_item: TaskItem,
        *,
        agent,
        worker_context: ExecutionContractContext,
    ):
        _ = agent
        _ = worker_context
        severity = str(task_item.payload.get("severity", "medium"))
        owner = str(task_item.payload.get("owner", "oncall-manager"))
        return {
            "severity": severity,
            "owner": owner,
            "action": "page-and-mitigate",
        }
