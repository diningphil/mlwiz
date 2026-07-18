(() => {
  "use strict";

  const optionStorageKey = "mlwiz-plot-code-options-v1";
  const bundleCapabilities = {
    aaai2024: { column: true },
    aistats2025: { column: true },
    colm2026: { relWidth: true, usetex: true },
    cvpr2024: { column: true, usetex: true },
    eccv2024: { relWidth: true },
    iclr2024: { relWidth: true, usetex: true },
    icml2024: { column: true, usetex: true },
    jmlr2001: { relWidth: true },
    neurips2024: { relWidth: true, usetex: true },
    probnum2025: { column: true },
    tmlr2023: { relWidth: true },
    tue_ai_thesis: { relWidth: true },
    uai2023: { column: true },
  };
  const defaultOptions = {
    bundle: "neurips2024",
    palette: "paultol_muted",
    width: "half",
    format: "pdf",
    latex: false,
    grid: true,
    title: true,
    legend: true,
  };
  let activeSpec = null;
  let initialized = false;

  function readOptions() {
    try {
      return { ...defaultOptions, ...JSON.parse(localStorage.getItem(optionStorageKey) || "{}") };
    } catch (_error) {
      return { ...defaultOptions };
    }
  }

  function pythonBoolean(value) {
    return value ? "True" : "False";
  }

  function cleanFileStem(value) {
    const stem = String(value || "mlwiz_plot")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 72);
    return stem || "mlwiz_plot";
  }

  function bundleExpression(options) {
    const capabilities = bundleCapabilities[options.bundle] || {};
    const arguments_ = [];
    if (capabilities.column) arguments_.push(`column=${JSON.stringify(options.width)}`);
    if (capabilities.relWidth) arguments_.push(`rel_width=${options.width === "full" ? "1.0" : "0.5"}`);
    if (capabilities.usetex) arguments_.push("usetex=USE_LATEX");
    return `bundles.${options.bundle}(${arguments_.join(", ")})`;
  }

  function commonPreamble(spec, options, needs3D = false, extraImports = "") {
    const serialized = JSON.stringify(spec, null, 2)
      .replaceAll('"""', "\\u0022\\u0022\\u0022");
    return `"""Reproduce an MLWiz dashboard plot with matplotlib and tueplots.

Install dependencies with:
    python -m pip install matplotlib numpy tueplots
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, cycler
from tueplots.constants.color import palettes
${needs3D ? "from mpl_toolkits.mplot3d import Axes3D  # noqa: F401\n" : ""}${extraImports}

DATA = json.loads(r"""${serialized}""")
USE_LATEX = ${pythonBoolean(options.latex)}

style = ${bundleExpression(options)}
# Some bundles intentionally fix a venue font; this explicit override keeps the
# LaTeX switch available for every selected publication style.
style["text.usetex"] = USE_LATEX
style.update(cycler.cycler(color=palettes.${options.palette}))
style.update({
    "axes.grid": ${pythonBoolean(options.grid)},
    "axes.grid.which": "both",
    "grid.alpha": 0.28,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
plt.rcParams.update(style)


def label(value):
    return value if isinstance(value, str) else json.dumps(value)


def numeric(values):
    return np.asarray([np.nan if value is None else float(value) for value in values])


def finite_values(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def adaptive_linear_threshold(values):
    magnitudes = np.abs(finite_values(values))
    magnitudes = magnitudes[magnitudes > 0]
    if not len(magnitudes):
        return 1.0
    return max(magnitudes.min(), magnitudes.max() * 1e-6, np.finfo(float).tiny)


def adaptive_log_transform(values, reference=None):
    values = np.asarray(values, dtype=float)
    reference = finite_values(values if reference is None else reference)
    if len(reference) and np.all(reference > 0):
        return np.log10(values)
    threshold = adaptive_linear_threshold(reference)
    scaled = np.abs(values) / threshold
    magnitude = np.where(scaled <= 1.0, scaled, 1.0 + np.log10(np.maximum(scaled, 1.0)))
    return np.sign(values) * magnitude


def set_adaptive_log_scale(axis, direction, values):
    values = finite_values(values)
    setter = getattr(axis, f"set_{direction}scale")
    if len(values) and np.all(values > 0):
        setter("log")
        return
    setter(
        "symlog", base=10, linthresh=adaptive_linear_threshold(values), linscale=0.9,
    )
`;
  }

  function finishingCode(spec, options, axisName = "ax") {
    const fileStem = cleanFileStem(spec.title);
    return `
${axisName}.set_title(DATA["title"] if ${pythonBoolean(options.title)} else "")
fig.savefig(${JSON.stringify(`${fileStem}.${options.format}`)}, bbox_inches="tight")
plt.show()
`;
  }

  function linePlotCode(spec, options) {
    return `${commonPreamble(spec, options)}

fig, ax = plt.subplots()
scale_values = []
for series in DATA["series"]:
    values = numeric(series["values"])
    positions = numeric(series.get("xValues") or list(range(1, len(values) + 1)))
    valid = np.isfinite(values)
    scale_values.extend(values[valid])
    color = ax._get_lines.get_next_color()
    raw_values = numeric(series.get("rawValues") or [])
    if len(raw_values):
        raw_valid = np.isfinite(raw_values)
        scale_values.extend(raw_values[raw_valid])
        (raw_line,) = ax.plot(
            positions[raw_valid], raw_values[raw_valid],
            color=color, linewidth=1, alpha=0.22, label="_nolegend_", zorder=1,
        )
        if series.get("dash"):
            raw_line.set_dashes(series["dash"])
    (line,) = ax.plot(
        positions[valid], values[valid], color=color, label=series["label"], zorder=2,
    )
    if series.get("dash"):
        line.set_dashes(series["dash"])
    if series.get("lower") and series.get("upper"):
        lower = numeric(series["lower"])
        upper = numeric(series["upper"])
        band_valid = np.isfinite(lower) & np.isfinite(upper)
        scale_values.extend(lower[band_valid])
        scale_values.extend(upper[band_valid])
        ax.fill_between(positions, lower, upper, where=band_valid, color=line.get_color(), alpha=0.16)

ax.set_xlabel(DATA.get("xLabel", "epoch"))
ax.set_ylabel(DATA.get("yLabel", "value"))
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", scale_values)
if ${pythonBoolean(options.legend)}:
    ax.legend()
${finishingCode(spec, options)}`;
  }

  function trend3DCode(spec, options) {
    return `${commonPreamble(spec, options, true, "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n")}

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
secondary_values = []
scale_values = []
for series in DATA["series"]:
    if series["secondary"] not in secondary_values:
        secondary_values.append(series["secondary"])
secondary_values.sort(key=lambda value: (not isinstance(value, (int, float)), label(value)))

for series in DATA["series"]:
    values = numeric(series["values"])
    positions = numeric(series.get("xValues") or list(range(1, len(values) + 1)))
    z_value = secondary_values.index(series["secondary"])
    valid = np.isfinite(values)
    scale_values.extend(values[valid])
    color = ax._get_lines.get_next_color()
    ax.plot(positions[valid], values[valid], np.full(valid.sum(), z_value), color=color, label=series["label"])
    if series.get("lower") and series.get("upper"):
        lower = numeric(series["lower"])
        upper = numeric(series["upper"])
        band_valid = np.isfinite(lower) & np.isfinite(upper)
        scale_values.extend(lower[band_valid])
        scale_values.extend(upper[band_valid])
        polygon = [
            *[(x, y, z_value) for x, y in zip(positions[band_valid], lower[band_valid])],
            *[(x, y, z_value) for x, y in zip(positions[band_valid][::-1], upper[band_valid][::-1])],
        ]
        if len(polygon) >= 3:
            ax.add_collection3d(Poly3DCollection([polygon], facecolor=color, alpha=0.12, edgecolor="none"))

ax.set_xlabel(DATA.get("xLabel", "epoch"))
ax.set_ylabel(DATA.get("yLabel", "value"))
ax.set_zlabel(DATA.get("zLabel", "group"))
ax.set_zticks(range(len(secondary_values)), [label(value) for value in secondary_values])
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", scale_values)
if ${pythonBoolean(options.legend)}:
    ax.legend()
${finishingCode(spec, options)}`;
  }

  function trajectory3DCode(spec, options) {
    return `${commonPreamble(spec, options, true)}

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
left_scale_values = []
right_scale_values = []
for series in DATA["series"]:
    left = numeric(series["leftValues"])
    right = numeric(series["rightValues"])
    point_count = min(len(left), len(right))
    positions = numeric(series.get("xValues") or list(range(1, point_count + 1)))[:point_count]
    valid = np.isfinite(left[:point_count]) & np.isfinite(right[:point_count])
    left_scale_values.extend(left[:point_count][valid])
    right_scale_values.extend(right[:point_count][valid])
    ax.plot(positions[valid], left[:point_count][valid], right[:point_count][valid], label=series["label"])

ax.set_xlabel(DATA.get("xLabel", "epoch"))
ax.set_ylabel(DATA.get("yLabel", "value"))
ax.set_zlabel(DATA.get("zLabel", "value"))
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", left_scale_values)
    set_adaptive_log_scale(ax, "z", right_scale_values)
if ${pythonBoolean(options.legend)}:
    ax.legend()
${finishingCode(spec, options)}`;
  }

  function barPlotCode(spec, options) {
    return `${commonPreamble(spec, options)}

fig, ax = plt.subplots()
x = np.arange(len(DATA["series"]))
means = np.asarray([series["mean"] for series in DATA["series"]], dtype=float)
stds = np.asarray([series["std"] for series in DATA["series"]], dtype=float)
ax.bar(x, means, yerr=stds, capsize=3)
ax.set_xticks(x, [label(series["primary"]) for series in DATA["series"]], rotation=30, ha="right")
ax.set_xlabel(DATA.get("xLabel", "hyperparameter"))
ax.set_ylabel(DATA.get("yLabel", "metric"))
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", np.concatenate([means, means - stds, means + stds]))
${finishingCode(spec, options)}`;
  }

  function violinPlotCode(spec, options) {
    return `${commonPreamble(spec, options)}

fig, ax = plt.subplots()
samples = [numeric(series["samples"]) for series in DATA["series"]]
parts = ax.violinplot(samples, showmeans=True, showextrema=True)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for index, body in enumerate(parts["bodies"]):
    body.set_facecolor(colors[index % len(colors)])
    body.set_edgecolor(colors[index % len(colors)])
    body.set_alpha(0.32)
if DATA.get("showPoints"):
    rng = np.random.default_rng(0)
    for index, values in enumerate(samples, start=1):
        ax.scatter(index + rng.uniform(-0.06, 0.06, len(values)), values, s=8, alpha=0.65)
ax.set_xticks(np.arange(1, len(DATA["series"]) + 1), [label(series["primary"]) for series in DATA["series"]], rotation=30, ha="right")
ax.set_xlabel(DATA.get("xLabel", "hyperparameter"))
ax.set_ylabel(DATA.get("yLabel", "metric"))
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", np.concatenate(samples) if samples else [])
${finishingCode(spec, options)}`;
  }

  function bar3DCode(spec, options) {
    return `${commonPreamble(spec, options, true)}

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
primary_values = []
secondary_values = []
for series in DATA["series"]:
    if series["primary"] not in primary_values:
        primary_values.append(series["primary"])
    if series["secondary"] not in secondary_values:
        secondary_values.append(series["secondary"])

heights = np.asarray([series["mean"] for series in DATA["series"]], dtype=float)
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    color_values = adaptive_log_transform(heights)
else:
    color_values = heights
normalizer = plt.Normalize(np.nanmin(color_values), np.nanmax(color_values) if np.nanmax(color_values) != np.nanmin(color_values) else np.nanmin(color_values) + 1)
for series, height, color_value in zip(DATA["series"], heights, color_values):
    x = primary_values.index(series["primary"])
    z = secondary_values.index(series["secondary"])
    ax.bar3d(x - 0.35, z - 0.35, 0, 0.7, 0.7, height, color=plt.cm.viridis(normalizer(color_value)), shade=True)

ax.set_xlabel(DATA.get("xLabel", "hyperparameter"))
ax.set_ylabel(DATA.get("zLabel", "second hyperparameter"))
ax.set_zlabel(DATA.get("yLabel", "metric"))
ax.set_xticks(range(len(primary_values)), [label(value) for value in primary_values])
ax.set_yticks(range(len(secondary_values)), [label(value) for value in secondary_values])
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "z", heights)
${finishingCode(spec, options)}`;
  }

  function violin3DCode(spec, options) {
    return `${commonPreamble(spec, options, true, "from matplotlib.collections import PolyCollection\n")}

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
primary_values = []
secondary_values = []
for series in DATA["series"]:
    if series["primary"] not in primary_values:
        primary_values.append(series["primary"])
    if series["secondary"] not in secondary_values:
        secondary_values.append(series["secondary"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
scale_values = []

for index, series in enumerate(DATA["series"]):
    values = numeric(series["samples"])
    values = values[np.isfinite(values)]
    if not len(values):
        continue
    scale_values.extend(values)
    x = primary_values.index(series["primary"])
    z = secondary_values.index(series["secondary"])
    spread = max(float(np.ptp(values)), np.finfo(float).eps)
    bandwidth = max(spread / max(2, np.sqrt(len(values))), spread / 40)
    y_grid = np.linspace(values.min() - 0.1 * spread, values.max() + 0.1 * spread, 70)
    density = np.exp(-0.5 * ((y_grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(axis=1)
    density = 0.32 * density / max(density.max(), np.finfo(float).eps)
    vertices = [*zip(x - density, y_grid), *zip((x + density)[::-1], y_grid[::-1])]
    collection = PolyCollection([vertices], facecolor=colors[index % len(colors)], edgecolor=colors[index % len(colors)], alpha=0.32)
    ax.add_collection3d(collection, zs=z, zdir="z")
    if DATA.get("showPoints"):
        jitter = np.random.default_rng(index).uniform(-0.05, 0.05, len(values))
        ax.scatter(x + jitter, values, np.full(len(values), z), s=7, alpha=0.65)

ax.set_xlabel(DATA.get("xLabel", "hyperparameter"))
ax.set_ylabel(DATA.get("yLabel", "metric"))
ax.set_zlabel(DATA.get("zLabel", "second hyperparameter"))
ax.set_xticks(range(len(primary_values)), [label(value) for value in primary_values])
ax.set_zticks(range(len(secondary_values)), [label(value) for value in secondary_values])
if DATA.get("scale") in ("log", "log-modulus", "symlog"):
    set_adaptive_log_scale(ax, "y", scale_values)
${finishingCode(spec, options)}`;
  }

  function generatePython(spec, options = {}) {
    const resolved = { ...defaultOptions, ...options };
    const generators = {
      line: linePlotCode,
      trend3d: trend3DCode,
      trajectory3d: trajectory3DCode,
      bar: barPlotCode,
      violin: violinPlotCode,
      bar3d: bar3DCode,
      violin3d: violin3DCode,
    };
    const generator = generators[spec.kind];
    if (!generator) throw new Error(`Unsupported plot kind: ${spec.kind}`);
    return generator(spec, resolved).trimStart();
  }

  function selectedOptions() {
    return {
      bundle: document.getElementById("plot-code-bundle").value,
      palette: document.getElementById("plot-code-palette").value,
      width: document.getElementById("plot-code-width").value,
      format: document.getElementById("plot-code-format").value,
      latex: document.getElementById("plot-code-latex").checked,
      grid: document.getElementById("plot-code-grid").checked,
      title: document.getElementById("plot-code-title-toggle").checked,
      legend: document.getElementById("plot-code-legend").checked,
    };
  }

  function syncPreview() {
    if (!activeSpec) return;
    const options = selectedOptions();
    document.getElementById("plot-code-preview").value = generatePython(activeSpec, options);
    try {
      localStorage.setItem(optionStorageKey, JSON.stringify(options));
    } catch (_error) {
      // Export still works if persistent storage is unavailable.
    }
  }

  function initialize() {
    if (initialized || typeof document === "undefined") return;
    const dialog = document.getElementById("plot-code-dialog");
    if (!dialog) return;
    initialized = true;
    const options = readOptions();
    document.getElementById("plot-code-bundle").value = options.bundle;
    document.getElementById("plot-code-palette").value = options.palette;
    document.getElementById("plot-code-width").value = options.width;
    document.getElementById("plot-code-format").value = options.format;
    document.getElementById("plot-code-latex").checked = options.latex;
    document.getElementById("plot-code-grid").checked = options.grid;
    document.getElementById("plot-code-title-toggle").checked = options.title;
    document.getElementById("plot-code-legend").checked = options.legend;
    dialog.querySelectorAll("select, input").forEach((control) => {
      control.addEventListener("change", syncPreview);
    });
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close("cancel");
    });
    document.getElementById("plot-code-copy").addEventListener("click", async () => {
      const preview = document.getElementById("plot-code-preview");
      try {
        await navigator.clipboard.writeText(preview.value);
      } catch (_error) {
        preview.select();
        document.execCommand("copy");
      }
      const status = document.getElementById("plot-code-status");
      status.textContent = "Copied to clipboard";
      setTimeout(() => { status.textContent = ""; }, 1600);
    });
    document.getElementById("plot-code-download").addEventListener("click", () => {
      const code = document.getElementById("plot-code-preview").value;
      const link = document.createElement("a");
      const url = URL.createObjectURL(new Blob([code], { type: "text/x-python;charset=utf-8" }));
      link.href = url;
      link.download = `${cleanFileStem(activeSpec?.title)}.py`;
      link.click();
      URL.revokeObjectURL(url);
    });
  }

  function open(specOrFactory) {
    initialize();
    activeSpec = typeof specOrFactory === "function" ? specOrFactory() : specOrFactory;
    if (!activeSpec) return;
    const dialog = document.getElementById("plot-code-dialog");
    document.getElementById("plot-code-description").textContent = `${activeSpec.title} · configure a publication-ready script.`;
    document.getElementById("plot-code-status").textContent = "";
    syncPreview();
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  function createButton(specOrFactory) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "plot-code-button";
    button.textContent = "</> Python";
    button.title = "Export Python code with tueplots";
    button.setAttribute("aria-label", "Export this plot as Python code");
    button.addEventListener("click", () => open(specOrFactory));
    return button;
  }

  globalThis.MLWizPlotExport = { createButton, generatePython, open };
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
    else initialize();
  }
})();
