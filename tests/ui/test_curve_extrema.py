"""Execute extrema calculations, canvas annotations, and generated plot scripts."""

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ASSETS = Path(__file__).parents[2] / "mlwiz" / "ui" / "web_assets"


def run_node(script):
    """Run the dashboard's JavaScript with Node, without a browser dependency."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard JavaScript tests")
    return subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout


def test_extrema_values_positions_gaps_ties_and_constants():
    run_node(f"""
const assert = require('node:assert/strict');
require({json.dumps(str(ASSETS / "curve_extrema.js"))});
const {{find, paired, summary}} = MLWizCurveExtrema;
assert.deepEqual(paired([2, null, -1, 4], [3, 9, null, 5]), [2, null, null, 4]);
assert.deepEqual(find([]), []);
assert.deepEqual(find([null, NaN, Infinity, -Infinity]), []);
assert.deepEqual(find([3, null, -2, 9, -2, 9], [10, 20, 30, 40, 50, 60]), [
  {{value: -2, index: 2, x: 30, kind: 'min'}},
  {{value: 9, index: 3, x: 40, kind: 'max'}},
]);
assert.deepEqual(find([5, 5]), [{{value: 5, index: 0, x: 1, kind: 'min = max'}}]);
assert.deepEqual(find([7]), [{{value: 7, index: 0, x: 1, kind: 'min = max'}}]);
assert.deepEqual(find([0, 2], [NaN, 42]), [{{value: 2, index: 1, x: 42, kind: 'min = max'}}]);
assert.equal(summary([0, -3, 2], String), 'min -3 · max 2');
assert.equal(summary([null], String), '');
assert.equal(find(Array.from({{length: 200000}}, (_, i) => i))[1].value, 199999);
""")


def test_canvas_extrema_use_projection_focus_and_visible_bounds():
    run_node(f"""
const assert = require('node:assert/strict');
require({json.dumps(str(ASSETS / "curve_extrema.js"))});
globalThis.document = {{documentElement: {{}}}};
globalThis.getComputedStyle = () => ({{getPropertyValue: () => '#fff'}});
const texts = [];
const opacities = [];
const ctx = {{save() {{}}, restore() {{}}, beginPath() {{}}, setLineDash() {{}},
  arc() {{}}, fill() {{opacities.push(this.globalAlpha);}},
  measureText(text) {{return {{width: text.length * 4}};}}, strokeText() {{}},
  fillText(text, x, y) {{texts.push({{text, x, y}});}}}};
const options = {{color: 'red', opacity: 0.2, format: String,
  bounds: {{left: 0, right: 200, top: 0, bottom: 100}}, xValues: [10, 20, 30]}};
const points = MLWizCurveExtrema.draw(ctx, [-2, 4, 1],
  (index, value) => ({{x: (index + 1) * 50, y: 50 - value * 5}}), options);
assert.deepEqual(points.map(p => [p.kind, p.x, p.y]), [['min', 50, 60], ['max', 100, 30]]);
assert.deepEqual(texts.map(p => p.text), ['min -2', 'max 4']);
assert.deepEqual(opacities, [0.2, 0.2]);
assert(texts.every(p => p.x >= 0 && p.x <= 200 && p.y >= 0 && p.y <= 100));
assert.equal(MLWizCurveExtrema.draw(ctx, [1, 2], () => ({{x: -10, y: 50}}), options).length, 0);
""")


@pytest.mark.parametrize("kind", ["line", "trend3d", "trajectory3d"])
@pytest.mark.parametrize("enabled", [False, True])
def test_generated_python_extrema_match_displayed_values(
    tmp_path, monkeypatch, kind, enabled
):
    series = {
        "label": "Validation",
        "values": [2, None, -1, 4],
        "rawValues": [8, None, -5, 9],
        "xValues": [10, 20, 30, 40],
        "secondary": "Group A",
        "leftValues": [2, None, -1, 4],
        "rightValues": [3, 9, None, 5],
    }
    spec = {
        "kind": kind,
        "title": "Extrema",
        "scale": "log",
        "smoothing": 0.5,
        "showExtrema": enabled,
        "series": [series],
        "yLabel": "Loss",
        "zLabel": "Score",
    }
    script = run_node(f"""
require({json.dumps(str(ASSETS / "plot_export.js"))});
process.stdout.write(MLWizPlotExport.generatePython({json.dumps(spec)}));
""")
    ast.parse(script)
    pytest.importorskip("tueplots")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        exec(compile(script, "dashboard_plot.py", "exec"), {})
        labels = [text.get_text() for text in plt.gcf().axes[0].texts]
        if not enabled:
            assert labels == []
        elif kind == "trajectory3d":
            assert labels == ["Loss min 2", "Loss max 4", "Score min 3", "Score max 5"]
        else:
            assert labels == ["min -1", "max 4"]
        assert list(tmp_path.glob("*.pdf"))
    finally:
        plt.close("all")


def test_controls_persistence_and_curve_renderer_integration():
    page = (ASSETS / "index.html").read_text()
    app = (ASSETS / "app.js").read_text()
    assert page.index("/assets/curve_extrema.js") < page.index("/assets/app.js")
    for control in ("curve-extrema", "analysis-curve-extrema"):
        assert f'id="{control}" type="checkbox"' in page
    assert "showExtrema: Boolean(storedState.showExtrema)" in app
    assert "showExtrema: state.showExtrema" in app
    assert "state.showExtrema = event.target.checked" in app
    assert "el(id).checked = state.showExtrema" in app
    assert (
        app.count("drawCurveExtrema(ctx,") == 5
    )  # 2D, 3D trend, two trajectory axes, helper
    assert "curveExtremaSummary(line.values" in app
    assert (
        "curveExtremaSummary(MLWizCurveExtrema.paired(line.leftValues, line.rightValues)"
        in app
    )
    assert (
        "curveExtremaSummary(MLWizCurveExtrema.paired(line.rightValues, line.leftValues)"
        in app
    )
