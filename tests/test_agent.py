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
    agent = GeoVLAAgent(bbox=BBOX, client=NS(messages=fake), use_llm=True)
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
    agent = GeoVLAAgent(bbox=BBOX, client=NS(messages=FakeMessages(looping)), use_llm=True)
    result = agent.run("elevation?", max_turns=3)
    assert "maximum number" in result["answer"]
