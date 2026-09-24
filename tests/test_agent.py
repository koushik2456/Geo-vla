from types import SimpleNamespace as NS

from agent import GeoVLAAgent
from planner import OfflinePlanner
from datetime import date

BBOX = [77.50, 12.90, 77.56, 12.96]


# -- offline planner ------------------------------------------------------------

def test_planner_flagship_query():
    plan = OfflinePlanner(today=date(2026, 9, 24)).plan(
        "Show me where deforestation increased near rivers in the last 3 years "
        "and estimate terrain slope in those zones.")
    tools = [s.tool for s in plan.steps]
    assert plan.dates == ("2023-07-01", "2026-07-01")
    assert tools[:2] == ["fetch_sentinel2_scene"] * 2
    for t in ("classify_scene", "class_mask", "overlay_layers", "fetch_osm_features",
              "buffer_features", "fetch_dem", "compute_slope", "zonal_stats"):
        assert t in tools
    assert plan.zone == "forest_loss_near_water"


def test_planner_parses_dates_and_distances():
    p = OfflinePlanner(today=date(2026, 9, 24))
    assert p.plan("changes between 2018-05-01 and 2024-05-01").dates == ("2018-05-01", "2024-05-01")
    assert p.plan("urban growth from 2017 to 2023").dates == ("2017-07-01", "2023-07-01")
    plan = p.plan("NDVI within 2 km of roads")
    buf = next(s for s in plan.steps if s.tool == "buffer_features")
    assert buf.input == {"layer_id": "osm_roads", "distance_m": 2000}


def test_offline_agent_end_to_end():
    result = GeoVLAAgent(bbox=BBOX).run("Where did forest loss happen near rivers since 2019? Include slope.")
    assert result["planner"] == "offline"
    assert "Forest loss" in result["answer"] and "slope" in result["answer"]
    calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    assert calls and not any(c["is_error"] for c in calls)
    assert any(layer["id"] == "forest_loss" for layer in result["layers"])


# -- Claude tool-use loop with a scripted fake client ------------------------------

def _block(type_, **kw):
    return NS(type=type_, **kw)


def _response(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content), usage=NS(input_tokens=100, output_tokens=20))


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        # Snapshot: the agent keeps appending to the same messages list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_llm_loop_executes_tools_and_returns_answer():
    fake = FakeMessages([
        _response("tool_use",
                  _block("thinking", thinking="Need two dates, then change detection."),
                  _block("tool_use", id="t1", name="fetch_sentinel2_scene", input={"date": "2020-07-01"}),
                  _block("tool_use", id="t2", name="fetch_sentinel2_scene", input={"date": "2024-07-01"})),
        _response("tool_use",
                  _block("tool_use", id="t3", name="detect_change",
                         input={"scene_id_1": "s2_2020-07-01", "scene_id_2": "s2_2024-07-01"}),
                  _block("tool_use", id="t4", name="compute_ndvi", input={"scene_id": "s2_1900-01-01"})),
        _response("end_turn", _block("text", text="About 4% of the area changed.")),
    ])
    agent = GeoVLAAgent(bbox=BBOX, client=NS(messages=fake), use_llm=True, provider="anthropic")
    result = agent.run("What changed between 2020 and 2024?")

    assert result["answer"] == "About 4% of the area changed."
    assert result["planner"] == "claude"
    assert result["usage"] == {"input_tokens": 300, "output_tokens": 60}
    assert [t["type"] for t in result["trace"]][:1] == ["thinking"]

    # Parallel tool calls come back in a single user message, errors flagged.
    second_request = fake.calls[1]["messages"]
    results = second_request[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    third = fake.calls[2]["messages"][-1]["content"]
    assert third[0]["is_error"] is False and third[1]["is_error"] is True
    assert "unknown layer" in third[1]["content"]
    # The bbox and date context reach the model in the first user turn.
    assert str(BBOX) in fake.calls[0]["messages"][0]["content"]
    assert len(fake.calls[0]["tools"]) == 15


def test_llm_loop_respects_max_turns():
    looping = [_response("tool_use", _block("tool_use", id=f"t{i}", name="fetch_dem", input={}))
               for i in range(3)]
    agent = GeoVLAAgent(bbox=BBOX, client=NS(messages=FakeMessages(looping)), use_llm=True, provider="anthropic")
    result = agent.run("elevation?", max_turns=3)
    assert "maximum number" in result["answer"]


# -- OpenAI-compatible providers (Groq, OpenRouter, Together, Ollama) --------------------------------

def _call(i, name, args):
    return NS(id=f"call_{i}", type="function", function=NS(name=name, arguments=args))


def _chat(content=None, calls=None, finish="tool_calls"):
    msg = NS(content=content, tool_calls=calls)
    return NS(choices=[NS(message=msg, finish_reason=finish if calls else "stop")],
              usage=NS(prompt_tokens=50, completion_tokens=10))


class FakeCompletions:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _groq_agent(responses):
    fake = FakeCompletions(responses)
    agent = GeoVLAAgent(bbox=BBOX, client=NS(chat=NS(completions=fake)), use_llm=True, provider="groq")
    return agent, fake


def test_openai_compatible_loop():
    agent, fake = _groq_agent([
        _chat("<think>Need imagery for two dates.</think>Fetching scenes.",
              [_call(1, "fetch_sentinel2_scene", '{"date": "2020-07-01"}'),
               _call(2, "fetch_sentinel2_scene", '{"date": "2024-07-01"}')]),
        _chat(None, [_call(3, "detect_change", '{"scene_id_1": "s2_2020-07-01", "scene_id_2": "s2_2024-07-01"}'),
                     _call(4, "compute_ndvi", '{"scene_id": oops}')]),
        _chat("About 4% changed."),
    ])
    result = agent.run("What changed?")
    assert result["answer"] == "About 4% changed." and result["planner"] == "llm"
    assert result["usage"] == {"input_tokens": 150, "output_tokens": 30}
    kinds = [t["type"] for t in result["trace"]]
    assert kinds[:2] == ["thinking", "text"]
    first = fake.calls[0]
    assert first["messages"][0]["role"] == "system" and first["tools"][0]["type"] == "function"
    assert {t["function"]["name"] for t in first["tools"]} >= {"fetch_sentinel2_scene", "overlay_layers"}
    third = fake.calls[2]["messages"]
    tool_msgs = [m for m in third if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3", "call_4"]
    assert "invalid JSON" in tool_msgs[-1]["content"]
    assert "change_2020-07-01_2024-07-01" in agent.workspace.layers


def test_openai_compatible_recovers_from_tool_use_failed():
    agent, fake = _groq_agent([Exception("Error code: 400 - tool_use_failed"), _chat("Done.")])
    assert agent.answer("hi")["answer"] == "Done."
    assert "malformed" in fake.calls[1]["messages"][-1]["content"]


def test_provider_resolution(monkeypatch):
    import config
    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(config, "PROVIDER_PRESETS", {**config.PROVIDER_PRESETS,
                        "groq": {**config.PROVIDER_PRESETS["groq"], "key": lambda: "gsk_test"}})
    s = config.llm_settings()
    assert s["provider"] == "groq" and s["base_url"].startswith("https://api.groq.com") and s["model"]
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    assert config.llm_settings() is None
