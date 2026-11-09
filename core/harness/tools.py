"""
================================================================================
BLOCK: Tool Contract & Registry (structured tool-calling with Pydantic)
--------------------------------------------------------------------------------
Role & contracts
  Every capability an agent can exercise is a ToolSpec:
    name, description, args_model (Pydantic, extra="forbid"), fn(args, ctx),
    privilege (read | write | egress | exec), idempotent, untrusted_output,
    timeout_s.
  Upstream : agent packages (Phase 2-5) register tools.
  Downstream: runner.AgentRunner (validation + guards + execution) and the LLM
             provider layer (json_schema() -> function-calling schemas).

Key engineering mechanism
  * StrictArgs base model: extra="forbid" + bounded string length -> smuggled
    fields ("bypass_cap": true) are a hard, non-switchable checkpoint.
  * Argument *kinds* (sql / path / email / url / percent / usd) are declared
    once on the field via ArgKind(...) metadata; Layer-1 policy binds rules to
    kinds, not to tool names -> a new tool inherits every rule for free.
  * Privilege + untrusted_output flags drive Layer-3 (taint & approvals).

Failure modes & 1-step mitigation
  * LLM hallucinates a tool -> registry miss = checkpoint block, the error is
    returned to the model as a tool message (agent can self-correct).
  * Malformed / extra args -> Pydantic ValidationError = checkpoint block.
  * Tool raises CheckpointBlocked (sandbox, DuckDB external access) -> runner
    records a non-switchable checkpoint decision instead of a crash.
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class Privilege(str, Enum):
    READ = "read"
    WRITE = "write"     # mutates state we own (flags, budgets, files)
    EGRESS = "egress"   # sends data outside the trust boundary (email, HTTP)
    EXEC = "exec"       # runs code

    @property
    def is_side_effecting(self) -> bool:
        return self is not Privilege.READ


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=20_000)


def ArgKind(kind: str, **kw) -> Any:
    """Field(...) carrying a semantic kind for Layer-1 argument rules."""
    return Field(json_schema_extra={"x-kind": kind}, **kw)


class TransientToolError(RuntimeError):
    """Retryable (e.g. upstream 503). Only retried when the tool is idempotent."""


class CheckpointBlocked(RuntimeError):
    """Raised by tool internals when a non-switchable checkpoint stops it."""

    def __init__(self, rule: str, detail: str = ""):
        self.rule, self.detail = rule, detail
        super().__init__(f"{rule}: {detail}")


@dataclass
class ToolContext:
    run_id: str
    agent: str
    defenses: Any  # redteam_guard.DefenseConfig
    extras: dict = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[StrictArgs]
    fn: Callable[[StrictArgs, ToolContext], Any]
    privilege: Privilege = Privilege.READ
    idempotent: bool = True
    untrusted_output: bool = False
    timeout_s: float = 5.0
    requires_approval: bool = False  # human/policy gate on EVERY call, tainted or not (L3)

    def arg_kinds(self) -> dict[str, str]:
        out = {}
        for fname, finfo in self.args_model.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and "x-kind" in extra:
                out[fname] = extra["x-kind"]
        return out

    def json_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        for prop in schema.get("properties", {}).values():
            prop.pop("x-kind", None)
        schema["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "parameters": schema}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {}
        for s in specs or []:
            self.register(s)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        return [self._specs[n].json_schema() for n in (names or self.names()) if n in self._specs]
