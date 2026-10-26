"""Keep component vector lengths out of dashboard epoch accounting."""

import json
from pathlib import Path

import pytest
import torch

from mlwiz.ui.dashboard import ResultsRepository, _recorded_epoch_count
from tests.ui.test_curve_extrema import run_node

APP = Path(__file__).parents[2] / "mlwiz/ui/web_assets/app.js"


def javascript_functions(*names):
    """Extract actual dashboard functions for execution with a small DOM stub."""
    source = APP.read_text()
    functions = []
    for name in names:
        start = source.index(f"  function {name}(")
        end = source.index("\n  function ", start + 1)
        functions.append(source[start:end])
    return "\n".join(functions)


@pytest.mark.parametrize("epochs", [1, 2, 6])
@pytest.mark.parametrize("layers", [1, 2, 5])
def test_layer_values_do_not_inflate_epochs_or_step_boundaries(
    tmp_path, epochs, layers
):
    run = tmp_path / "final_run1"
    run.mkdir()
    torch.save(
        {
            "losses": {"training_main_loss": [0.8] * epochs},
            "scores": {"validation_main_score": [0.7] * epochs},
            "model_widths": [16] * layers,
            "step": {
                "steps": [epochs * 10],
                "last_step": epochs * 10,
                "losses": {"training_main_loss": [0.5]},
            },
        },
        run / "metrics_data.torch",
    )
    details = ResultsRepository(tmp_path).details("final_run1")
    assert details["errors"] == []
    assert _recorded_epoch_count(details["series"]) == epochs
    step = next(item for item in details["series"] if item["unit"] == "step")
    assert [entry["epoch"] for entry in step["epoch_boundaries"]] == list(
        range(1, epochs + 1)
    )
    assert step["epoch_boundaries"][-1]["step"] == epochs * 10
    functions = javascript_functions(
        "seriesXValues", "recordedHistoryPosition", "renderDetails"
    )
    output = run_node(f"""
{functions}
let summary;
const state = {{details: {json.dumps(details)}, metricUnit: 'epoch'}};
const noop = () => {{}};
const renderCacheStatus = noop, renderExperimentOverview = noop,
  renderMetricUnit = noop, renderPlotMode = noop, renderInnerFoldAggregation = noop,
  renderOutlierFilter = noop, renderSourceFilter = noop, renderPlotNavigator = noop,
  renderCharts = noop, renderMetadata = noop, prepareModelGraph = noop;
const el = () => ({{}});
const readableName = name => name;
const formatTime = () => '';
const metricSeries = () => state.details.series.filter(item => item.unit === state.metricUnit);
const metricUnitLabel = capitalize => capitalize ? 'Epoch' : 'epoch';
const renderSummary = items => {{summary = items;}};
renderDetails();
process.stdout.write(JSON.stringify(summary));
""")
    assert ["Epochs recorded", epochs] in json.loads(output)


def test_epoch_positions_preserve_custom_histories_and_independent_sources():
    functions = javascript_functions(
        "seriesXValues", "recordedHistoryPosition"
    )
    run_node(f"""
const assert = require('node:assert/strict');
{functions}
const loss = {{source: 'run_1', group: 'losses', values: [.9, .8]}};
const widths = {{source: 'run_1', group: 'widths', values: [4, 8, 16, 32, 64]}};
assert.equal(recordedHistoryPosition([loss, widths], 'epoch'), 2);
assert.equal(recordedHistoryPosition([widths], 'epoch'), 5); // legacy custom-only histories
assert.equal(recordedHistoryPosition([{{...loss, values: []}}, widths], 'epoch'), 5);
assert.equal(recordedHistoryPosition([loss, widths, {{...loss, source: 'run_2', values: [1, 2, 3]}}], 'epoch'), 3);
assert.equal(recordedHistoryPosition([loss, widths, {{...widths, source: 'run_2', values: [1, 2, 3, 4, 5, 6]}}], 'epoch'), 6);
assert.equal(recordedHistoryPosition([{{group: 'losses', values: [1, 2], x_values: [10, 55]}}], 'step'), 55);
assert.equal(recordedHistoryPosition([], 'epoch'), 0);
""")


def test_epoch_layer_matrices_still_produce_one_curve_per_layer(tmp_path):
    run = tmp_path / "final_run1"
    run.mkdir()
    torch.save(
        {"model_widths": [[4, 8, 16, 32, 64], [5, 9, 17, 33, 65]]},
        run / "metrics_data.torch",
    )
    series = ResultsRepository(tmp_path).details("final_run1")["series"]
    assert len(series) == 5
    assert all(len(item["values"]) == 2 for item in series)
    assert _recorded_epoch_count(series) == 2
