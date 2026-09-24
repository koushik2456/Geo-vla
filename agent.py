"""
agent.py — Reasoning layer: an LLM-driven agent that decomposes a natural-
language geospatial instruction into a sequence of tool calls, executes them
via geotools.py, and returns a final synthesized answer plus the full
reasoning trace (used by the UI's transparency panel).

Uses the Anthropic API's native tool-use (function-calling) loop directly,
rather than a LangChain/LangGraph wrapper — this keeps the reasoning loop
fully visible and auditable, which matters for the "transparent reasoning"
requirement of the project, and avoids an extra framework dependency.

LLM providers (config.llm_settings): Groq, OpenRouter, Together, Ollama or any
OpenAI-compatible server hosting open models, or Anthropic. Without any key
the agent falls back to planner.OfflinePlanner,
a keyword-based planner that drives the same tools, so the demo always runs.
"""
import json
import logging
import re
import time
from datetime import date

import config
import geotools
from planner import OfflinePlanner

log = logging.getLogger("geo-vla.agent")

SYSTEM_PROMPT = """You are Geo-VLA, an autonomous geospatial analyst for remote sensing.

Given a natural-language instruction, plan and execute a chain of tool calls that answers it,
then give a concise final answer.

How the tools fit together:
- Data tools (fetch_sentinel2_scene, fetch_dem, fetch_osm_features) create layers with ids that
  later tools take as input. Fetch every date you need before analysing it.
- Scene analysis: compute_ndvi, classify_scene (land cover), detect_change (between two scenes).
- Terrain: compute_slope and flood_extent work on the DEM.
- Spatial reasoning: class_mask, threshold_layer, buffer_features, overlay_layers and zonal_stats
  turn intermediate results into zones and answer "where" and "how much" questions.
  For example, forest loss = overlay_layers(forest_<t1>, forest_<t2>, "difference").

Rules:
- Ground every number in your answer in actual tool outputs. Never invent values.
- If a tool returns an error, adapt (a different date window, a different layer) or explain the limitation.
- Mention which method produced key numbers when a tool reports a fallback or synthetic source.
- Keep the final answer short: the direct answer first, then 2-4 supporting figures (areas in km², fractions as %).
- If the user gives no dates, choose sensible ones relative to today's date. Prefer
  June–September acquisitions in the northern hemisphere (less cloud and snow).
- Use layer ids exactly as tools return them (e.g. "s2_2024-07-01", "landcover_2024-07-01").
- Call tools through function calls only, with valid JSON arguments. Omit bbox to use the selected area."""

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


class GeoVLAAgent:
    """Per-request agent. Holds the workspace of layers produced by tool
    calls and the ordered trace of reasoning, tool inputs and tool outputs."""

    def __init__(self, bbox=None, client=None, use_llm=None, on_event=None, provider=None):
        self.workspace = geotools.Workspace(bbox)
        self.trace: list = []
        self.llm = config.llm_settings()
        self.use_llm = (self.llm is not None) if use_llm is None else use_llm
        # Provider decides the wire format: Anthropic Messages or OpenAI-compatible chat completions.
        self.provider = provider or (self.llm["provider"] if self.llm else "anthropic")
        self.model = self.llm["model"] if self.llm else config.MODEL_NAME
        self._client = client
        self._on_event = on_event   # called with the trace after every new entry (live UI updates)

    def _log(self, entry: dict) -> None:
        self.trace.append(entry)
        if self._on_event:
            self._on_event(self.trace)

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.llm["api_key"])
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.llm["api_key"], base_url=self.llm["base_url"], timeout=120)
        return self._client

    # -- tool execution -----------------------------------------------------

    def execute_tool(self, name: str, tool_input: dict) -> tuple:
        """Run one tool, record it in the trace, return (output, is_error)."""
        start = time.perf_counter()
        try:
            output, is_error = geotools.run_tool(self.workspace, name, tool_input), False
        except (geotools.ToolError, ValueError, TypeError, KeyError) as exc:
            output, is_error = {"error": str(exc)}, True
        except Exception as exc:  # network / data-source failures surface to the LLM, not as a 500
            log.exception("tool %s failed", name)
            output, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
        self._log({
            "type": "tool_call", "tool": name, "input": tool_input, "output": output,
            "is_error": is_error, "duration_ms": round((time.perf_counter() - start) * 1000),
        })
        return output, is_error

    # -- main entrypoint ----------------------------------------------------

    def answer(self, instruction: str, max_turns: int = None) -> dict:
        """Plan and execute a free-text instruction. Layers stay in self.workspace."""
        if self.use_llm:
            answer, meta = self._run_llm(instruction, max_turns or config.MAX_AGENT_TURNS)
        else:
            answer, meta = self._run_offline(instruction)
        return {"answer": answer, "trace": self.trace, "data_mode": config.data_mode(), **meta}

    def run(self, instruction: str, max_turns: int = None) -> dict:
        """answer() plus every layer rendered inline (synchronous /query API)."""
        result = self.answer(instruction, max_turns)
        result["layers"] = [geotools.render_layer(layer) for layer in self.workspace.layers.values()]
        return result

    def run_plan(self, plan, rationale: str = None) -> dict:
        """Execute a fixed plan (sector workflows): same tools, same trace, no LLM.
        Returns {"results": {step key: output}, "failed": [step keys]}."""
        self._log({"type": "text", "text": rationale or plan.rationale})
        results, failed = {}, []
        for step in plan.steps:
            output, is_error = self.execute_tool(step.tool, step.input)
            if is_error:
                failed.append(step.key)
                if not step.optional:
                    raise geotools.ToolError(f"step '{step.tool}' failed: {output['error']}")
                continue
            results[step.key] = output
        return {"results": results, "failed": failed}

    def _context_message(self, instruction: str) -> str:
        bbox = self.workspace.default_bbox
        aoi = f"Selected area of interest (bbox): {bbox}" if bbox else "No area selected; ask for or infer a bbox."
        return f"Today's date: {date.today().isoformat()}\n{aoi}\n\nInstruction: {instruction}"

    def _run_llm(self, instruction: str, max_turns: int) -> tuple:
        if self.provider != "anthropic":
            return self._run_openai_compatible(instruction, max_turns)
        return self._run_anthropic(instruction, max_turns)

    # -- OpenAI-compatible providers: Groq, OpenRouter, Together, Ollama, vLLM, … -------------

    @staticmethod
    def _openai_tools() -> list:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}}
                for t in geotools.tool_schemas()]

    def _run_openai_compatible(self, instruction: str, max_turns: int) -> tuple:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._context_message(instruction)}]
        tools = self._openai_tools()
        usage = {"input_tokens": 0, "output_tokens": 0}
        meta = {"planner": "llm", "model": f"{self.provider}:{self.model}", "usage": usage}
        retries = 0

        for _ in range(max_turns):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto",
                    temperature=config.LLM_TEMPERATURE, max_tokens=4096)
            except Exception as exc:
                # Open models sometimes emit a malformed call that the provider rejects
                # (Groq: "tool_use_failed"). Nudge once or twice, then give up.
                if "tool_use_failed" in str(exc) and retries < 2:
                    retries += 1
                    self._log({"type": "text", "text": "Model produced an invalid tool call; asking it to retry."})
                    messages.append({"role": "user", "content": "Your last tool call was malformed. Call one tool "
                                     "at a time with valid JSON arguments matching the schema."})
                    continue
                raise
            if getattr(response, "usage", None):
                usage["input_tokens"] += response.usage.prompt_tokens or 0
                usage["output_tokens"] += response.usage.completion_tokens or 0
            msg = response.choices[0].message
            content = msg.content or ""
            for thought in _THINK_RE.findall(content):          # reasoning models (Qwen3, DeepSeek-R1)
                if thought.strip():
                    self._log({"type": "thinking", "text": thought.strip()})
            content = _THINK_RE.sub("", content).strip()
            calls = msg.tool_calls or []

            if not calls:
                if response.choices[0].finish_reason == "length":
                    content += "\n\n(Answer truncated: reached the token limit.)"
                return content, meta

            if content:
                self._log({"type": "text", "text": content})
            messages.append({"role": "assistant", "content": content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as exc:
                    output = {"error": f"invalid JSON arguments: {exc}"}
                    self._log({"type": "tool_call", "tool": c.function.name, "input": c.function.arguments,
                               "output": output, "is_error": True, "duration_ms": 0})
                else:
                    output, _ = self.execute_tool(c.function.name, args)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(output)})

        return "Reached the maximum number of reasoning turns without a final answer.", meta

    # -- Anthropic ---------------------------------------------------------------------------------

    def _run_anthropic(self, instruction: str, max_turns: int) -> tuple:
        messages = [{"role": "user", "content": self._context_message(instruction)}]
        request = {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "tools": geotools.tool_schemas(),
        }
        if config.THINKING_DISPLAY != "off":
            request["thinking"] = {"type": "adaptive", "display": config.THINKING_DISPLAY}
        usage = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(max_turns):
            response = self.client.messages.create(messages=messages, **request)
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            # Append the full content (thinking blocks included) so the next turn sees them unchanged.
            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "thinking" and getattr(block, "thinking", ""):
                    self._log({"type": "thinking", "text": block.thinking})
                elif block.type == "text" and block.text.strip() and response.stop_reason == "tool_use":
                    self._log({"type": "text", "text": block.text})

            meta = {"planner": "claude", "model": self.model, "usage": usage}
            if response.stop_reason == "refusal":
                return "The model declined this request.", meta
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n(Answer truncated: reached max_tokens.)"
                return text, meta

            # Execute every tool_use block, return all results in one user message.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output, is_error = self.execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(output), "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})

        return ("Reached the maximum number of reasoning turns without a final answer.",
                {"planner": "claude", "model": self.model, "usage": usage})

    def _run_offline(self, instruction: str) -> tuple:
        planner = OfflinePlanner(today=date.today())
        plan = planner.plan(instruction)
        self._log({"type": "text", "text": plan.rationale})
        results = {}
        for step in plan.steps:
            output, is_error = self.execute_tool(step.tool, step.input)
            results[step.key] = output
            if is_error:
                return (f"Stopped at step '{step.tool}': {output['error']}",
                        {"planner": "offline", "model": None})
        return planner.summarize(plan, results), {"planner": "offline", "model": None}
