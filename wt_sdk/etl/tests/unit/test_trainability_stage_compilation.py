"""Guard the trainability stage's integration into the ETL package.

Merging stage changes into the ETL must never introduce syntax, import, or
pipeline-wiring errors. These tests exercise that boundary without any
gateway access: the stage source is compiled explicitly, the stage and
pipeline modules are imported, the landing pipeline is assembled with the
stage wired in, and the sampling tool's offline session path runs the stage
end to end.
"""

import importlib
import json
import pkgutil
from pathlib import Path

from wt_sdk.etl import UpdateIsTrainableStage, load_pipeline
from wt_sdk.etl.tools.trainability_sample import process_session

_JOB_ID = "dataset#harness#model#task#20260911#owner#extra"
_SESSION_ID = "session-1"


def test_trainability_stage_source_compiles():
    stage_path = Path(__file__).parents[2] / "stages" / "trainability.py"
    compile(stage_path.read_text(encoding="utf-8"), str(stage_path), "exec")


def test_every_stage_and_pipeline_module_imports():
    for package_name in ("wt_sdk.etl.stages", "wt_sdk.etl.pipelines"):
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(
            package.__path__,
            prefix=f"{package_name}.",
        ):
            importlib.import_module(module_info.name)
    importlib.import_module("wt_sdk.etl.tools.trainability_sample")


def test_trainability_stage_wired_into_landing_pipeline():
    pipeline = load_pipeline("landing_enrichment_pipeline")

    trainability_stage = pipeline.ordered_stages[0]
    assert isinstance(trainability_stage, UpdateIsTrainableStage)
    assert trainability_stage.name == "update_is_trainable"
    assert trainability_stage.dependencies == ()
    assert pipeline.ordered_stages[1].dependencies == ("update_is_trainable",)


def test_trainability_stage_runs_inside_landing_pipeline():
    pipeline = load_pipeline("landing_enrichment_pipeline")

    result = pipeline.process_session(_completed_session())

    assert result.failures == ()
    assert result.successful_rows == 4
    patches = {patch.record_id: patch.updates for patch in result.landing_patches}
    assert patches["row-3"]["is_trainable"] is True
    assert patches["row-3"]["reward"] == 0.7
    assert "row-1" not in patches
    assert "row-2" not in patches
    assert "row-4" not in patches


def test_sample_tool_process_session_runs_stage_and_agrees_with_patches():
    report = process_session(_completed_session(), _JOB_ID, _SESSION_ID)

    assert report["status"] == "processed"
    assert report["warnings"] == []
    assert report["row_count"] == 4
    assert report["eligible_row_count"] == 4
    assert report["trainable_step_ids"] == [3]
    assert report["is_trainable_true_count"] == 1
    reason_codes = {
        row["id"]: row["trainability_diagnostics"]["reason_code"]
        for row in report["rows"]
    }
    assert reason_codes == {
        "row-1": "superseded_in_chain",
        "row-2": "superseded_in_chain",
        "row-3": "selected_chain_tail",
        "row-4": "short_side_chain",
    }


def _row(row_id: str, step_id: int, messages: list, **updates: object) -> dict:
    row = {
        "dataset_type": "trajectory",
        "dt": "2026-09-11",
        "id": row_id,
        "session_id": _SESSION_ID,
        "created_at": 1_758_000_000,
        "source_updated_at": 1_758_000_000_000,
        "serving_updated_at": None,
        "step_id": step_id,
        "is_terminal": False,
        "step_reward": None,
        "reward": None,
        "messages": json.dumps(messages),
        "response": json.dumps({"role": "assistant", "content": "answer"}),
        "chosen_trace": None,
        "rejected_trace": None,
        "ground_truth_answer": None,
        "reference_answer": None,
        "search_text": None,
        "agent_model": "opencode-model",
        "env_name": "test-env",
        "is_session_completed": False,
        "is_trainable": False,
        "meta_json": json.dumps({"provider_messages": []}),
        "tags": None,
        "env_id": "env-1",
        "job_id": _JOB_ID,
        "is_truncated": False,
        "blob_manifest": [],
    }
    row.update(updates)
    return row


def _completed_session() -> list[dict]:
    user_turn = {"role": "user", "content": "question"}
    assistant_turn = {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
    }
    return [
        _row("row-1", 0, [user_turn]),
        _row("row-2", 1, [user_turn, assistant_turn]),
        _row("row-4", 2, [{"role": "system", "content": "side conversation"}]),
        _row(
            "row-3",
            3,
            [user_turn, assistant_turn, {"role": "user", "content": "follow-up"}],
            is_session_completed=True,
            reward=0.7,
        ),
    ]
