(() => {
  "use strict";

  const state = {
    tree: null,
    details: null,
    selectedPath: null,
    group: "all",
    source: "all",
    query: "",
    treeQuery: "",
    charts: [],
  };

  const el = (id) => document.getElementById(id);
  const treeElement = el("experiment-tree");
  const detailsView = el("details-view");
  const welcome = el("welcome");
  const colors = {
    training: "#138a62",
    validation: "#4776e6",
    test: "#ef8354",
    other: "#8257e5",
  };

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function formatNumber(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    const absolute = Math.abs(number);
    if ((absolute > 0 && absolute < 0.001) || absolute >= 10000) return number.toExponential(3);
    return number.toLocaleString(undefined, { maximumFractionDigits: 5 });
  }

  function formatTime(timestamp) {
    if (!timestamp) return "No metric timestamp";
    const date = new Date(timestamp * 1000);
    return `Updated ${date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}`;
  }

  async function getJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  async function loadTree({ quiet = false } = {}) {
    const refresh = el("refresh-button");
    refresh.classList.add("loading");
    try {
      const tree = await getJson("/api/tree");
      state.tree = tree;
      el("root-path").textContent = tree.root;
      el("experiment-count").textContent = String(tree.experiment_count);
      el("last-refresh").textContent = `Refreshed ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      renderTree();
      if (state.selectedPath) await loadDetails(state.selectedPath, { preserveScroll: true, quiet: true });
    } catch (error) {
      if (!quiet) renderTreeError(error.message);
    } finally {
      refresh.classList.remove("loading");
    }
  }

  function renderTreeError(message) {
    treeElement.replaceChildren();
    const empty = node("div", "tree-empty");
    empty.append(node("strong", "", "Could not read results"), document.createTextNode(message));
    treeElement.append(empty);
  }

  function matchesTreeSearch(...values) {
    if (!state.treeQuery) return true;
    const text = values.filter(Boolean).join(" ").toLowerCase();
    return text.includes(state.treeQuery);
  }

  function renderTree() {
    treeElement.replaceChildren();
    if (!state.tree || state.tree.experiments.length === 0) {
      const empty = node("div", "tree-empty");
      empty.append(node("strong", "", "No MLWiz experiments found"));
      empty.append(document.createTextNode("Point --logdir to a folder containing MODEL_ASSESSMENT results."));
      treeElement.append(empty);
      return;
    }

    for (const experiment of state.tree.experiments) {
      const expMatches = matchesTreeSearch(experiment.name, experiment.path);
      const visibleFolds = experiment.outer_folds.filter((fold) => {
        if (expMatches || matchesTreeSearch(`outer fold ${fold.number}`)) return true;
        return fold.model_selection.some((config) =>
          matchesTreeSearch(`config ${config.number}`, JSON.stringify(config.results || {}))
        ) || fold.final_runs.some((run) => matchesTreeSearch(`final run ${run.number}`));
      });
      if (!visibleFolds.length && !expMatches) continue;

      const expDetails = node("details", "tree-group");
      expDetails.open = true;
      const summary = node("summary", "experiment-summary");
      summary.append(
        node("span", "chevron", "›"),
        node("span", "experiment-icon", "MW"),
        node("span", "", experiment.name),
        node("span", "tree-count", `${experiment.run_count} runs`)
      );
      expDetails.append(summary);

      for (const fold of visibleFolds) expDetails.append(renderFold(fold));
      treeElement.append(expDetails);
    }
  }

  function renderFold(fold) {
    const foldDetails = node("details", "fold-group");
    foldDetails.open = true;
    const summary = node("summary", "fold-summary");
    summary.append(node("span", "chevron", "›"), node("span", "", `Outer fold ${fold.number}`));
    foldDetails.append(summary);

    if (fold.model_selection.length) {
      foldDetails.append(node("div", "tree-section-label", "Model selection"));
      for (const config of fold.model_selection) {
        if (!matchesTreeSearch(`config ${config.number}`, JSON.stringify(config.results || {}), `outer fold ${fold.number}`)) continue;
        const configDetails = node("details", "config-group");
        const configSummary = node("summary", "config-summary");
        configSummary.append(node("span", "chevron", "›"));
        const configLink = node("span", "", `Configuration ${config.number}`);
        configSummary.append(configLink);
        if (config.is_winner) configSummary.append(node("span", "winner", "winner"));
        configSummary.addEventListener("click", (event) => {
          if (event.target === configSummary || event.target === configLink) {
            event.preventDefault();
            configDetails.open = !configDetails.open;
            loadDetails(config.path);
          }
        });
        configDetails.append(configSummary);
        for (const inner of config.inner_folds) {
          configDetails.append(node("div", "inner-label", `Inner fold ${inner.number}`));
          for (const run of inner.runs) configDetails.append(runButton(run, `Run ${run.number}`));
        }
        foldDetails.append(configDetails);
      }
    }

    if (fold.final_runs.length) {
      foldDetails.append(node("div", "tree-section-label", "Final runs"));
      for (const run of fold.final_runs) {
        if (matchesTreeSearch(`final run ${run.number}`, `outer fold ${fold.number}`)) {
          foldDetails.append(runButton(run, `Final run ${run.number}`));
        }
      }
    }
    return foldDetails;
  }

  function runButton(run, label) {
    const button = node("button", "tree-button", "");
    button.type = "button";
    button.dataset.path = run.path;
    if (!run.has_metrics) button.classList.add("no-metrics");
    if (run.path === state.selectedPath) button.classList.add("active");
    button.append(node("span", "metric-indicator"), node("span", "", label));
    button.title = run.has_metrics ? "Open epoch metrics" : "No metrics_data.torch yet";
    button.addEventListener("click", () => loadDetails(run.path));
    return button;
  }

  async function loadDetails(path, { preserveScroll = false, quiet = false } = {}) {
    const previousScroll = window.scrollY;
    state.selectedPath = path;
    renderTree();
    detailsView.hidden = false;
    welcome.hidden = true;
    if (!quiet) {
      el("selection-kind").textContent = "Loading metrics";
      el("selection-name").textContent = path.split("/").pop();
      el("selection-path").textContent = path;
      el("chart-grid").replaceChildren(chartMessage("Reading metrics_data.torch…"));
    }
    try {
      state.details = await getJson(`/api/details?path=${encodeURIComponent(path)}`);
      state.source = "all";
      renderDetails();
      if (!preserveScroll && window.innerWidth < 721) detailsView.scrollIntoView({ behavior: "smooth" });
      if (preserveScroll) window.scrollTo(0, previousScroll);
    } catch (error) {
      renderDetailsError(error.message);
    }
  }

  function chartMessage(message, title = "No charts to display") {
    const empty = node("div", "chart-empty");
    empty.append(node("strong", "", title), document.createTextNode(message));
    return empty;
  }

  function renderDetailsError(message) {
    el("chart-grid").replaceChildren(chartMessage(message, "Could not load this selection"));
  }

  function renderDetails() {
    const data = state.details;
    el("selection-kind").textContent = data.selection.kind;
    el("selection-name").textContent = readableName(data.selection.name);
    el("selection-path").textContent = data.selection.path;
    el("freshness").textContent = formatTime(data.modified_at);

    const maxEpochs = data.series.reduce((max, series) => Math.max(max, series.values.length), 0);
    const sources = [...new Set(data.series.map((series) => series.source))];
    renderSummary([
      ["Metric series", data.series.length],
      ["Epochs recorded", maxEpochs || "—"],
      ["Run files", data.metrics_file_count],
    ]);
    renderSourceFilter(sources);

    const notice = el("notice");
    if (data.errors.length) {
      notice.hidden = false;
      notice.className = "notice error";
      notice.textContent = `${data.errors.length} metric file${data.errors.length === 1 ? "" : "s"} could not be read. The remaining data is shown.`;
    } else if (!data.series.length) {
      notice.hidden = false;
      notice.className = "notice";
      notice.textContent = "No epoch histories were found here. Configure the Plotter callback, or wait for the run to write metrics_data.torch.";
    } else {
      notice.hidden = true;
    }

    renderCharts();
    renderMetadata(data.metadata);
  }

  function readableName(name) {
    const match = name.match(/^final_run_?(\d+)$/);
    if (match) return `Final run ${match[1]}`;
    const config = name.match(/^config_(\d+)$/);
    if (config) return `Configuration ${config[1]}`;
    const run = name.match(/^run_?(\d+)$/);
    if (run) return `Run ${run[1]}`;
    return name.replaceAll("_", " ");
  }

  function renderSummary(items) {
    const grid = el("summary-grid");
    grid.replaceChildren();
    for (const [label, value] of items) {
      const card = node("div", "summary-card");
      card.append(node("span", "", label), node("strong", "", String(value)));
      grid.append(card);
    }
  }

  function renderSourceFilter(sources) {
    const field = el("source-field");
    const select = el("source-select");
    select.replaceChildren();
    if (sources.length <= 1) {
      field.hidden = true;
      return;
    }
    field.hidden = false;
    const all = node("option", "", `All runs (${sources.length})`);
    all.value = "all";
    select.append(all);
    for (const source of sources) {
      const option = node("option", "", source);
      option.value = source;
      select.append(option);
    }
    select.value = state.source;
  }

  function splitMetric(name) {
    for (const split of ["training", "validation", "test"]) {
      if (name.startsWith(`${split}_`)) return { split, metric: name.slice(split.length + 1) };
    }
    return { split: "other", metric: name };
  }

  function groupedSeries() {
    const groups = new Map();
    for (const series of state.details.series) {
      if (state.group !== "all" && series.group !== state.group) continue;
      if (state.source !== "all" && series.source !== state.source) continue;
      const { split, metric } = splitMetric(series.name);
      if (state.query && !`${series.name} ${series.source} ${series.group}`.toLowerCase().includes(state.query)) continue;
      const key = `${series.source}\u0000${series.group}\u0000${metric}`;
      if (!groups.has(key)) groups.set(key, { source: series.source, group: series.group, metric, lines: [] });
      groups.get(key).lines.push({ split, name: series.name, values: series.values });
    }
    return [...groups.values()].sort((a, b) =>
      `${a.group} ${a.metric} ${a.source}`.localeCompare(`${b.group} ${b.metric} ${b.source}`)
    );
  }

  function renderCharts() {
    state.charts = [];
    const grid = el("chart-grid");
    grid.replaceChildren();
    const groups = groupedSeries();
    if (!groups.length) {
      grid.append(chartMessage("Try another metric type, run, or search term."));
      return;
    }
    for (const group of groups) {
      const card = node("article", "chart-card");
      const head = node("div", "chart-head");
      const title = node("div", "chart-title");
      title.append(node("h3", "", group.metric.replaceAll("_", " ")), node("p", "", group.source));
      head.append(title, node("span", "chart-type", group.group));
      const wrap = node("div", "chart-wrap");
      const canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", `${group.metric} over epochs`);
      wrap.append(canvas);
      const legend = node("div", "chart-legend");
      for (const line of group.lines) {
        const item = node("span", "legend-item");
        const swatch = node("span", "legend-swatch");
        swatch.style.background = colors[line.split] || colors.other;
        const valid = line.values.filter((value) => value !== null);
        item.append(swatch, document.createTextNode(`${line.split} `), node("span", "legend-value", formatNumber(valid.at(-1))));
        legend.append(item);
      }
      card.append(head, wrap, legend);
      grid.append(card);
      state.charts.push({ canvas, group });
      drawChart(canvas, group);
    }
  }

  function drawChart(canvas, group) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    const margin = { top: 12, right: 12, bottom: 27, left: 48 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = group.lines.flatMap((line) => line.values).filter((value) => value !== null && Number.isFinite(value));
    if (!values.length) return;
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) { min -= Math.abs(min || 1) * 0.05; max += Math.abs(max || 1) * 0.05; }
    const padding = (max - min) * 0.08;
    min -= padding;
    max += padding;
    const epochs = Math.max(...group.lines.map((line) => line.values.length), 1);
    const x = (index) => margin.left + (epochs <= 1 ? plotWidth / 2 : (index / (epochs - 1)) * plotWidth);
    const y = (value) => margin.top + ((max - value) / (max - min)) * plotHeight;

    ctx.font = '9px Inter, -apple-system, sans-serif';
    ctx.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = min + ((max - min) * tick) / 4;
      const tickY = y(value);
      ctx.beginPath(); ctx.moveTo(margin.left, tickY); ctx.lineTo(width - margin.right, tickY);
      ctx.strokeStyle = "#edf0ed"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#8a94a1"; ctx.textAlign = "right";
      ctx.fillText(formatNumber(value), margin.left - 7, tickY);
    }
    ctx.fillStyle = "#8a94a1"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("1", x(0), height - margin.bottom + 8);
    if (epochs > 1) ctx.fillText(String(epochs), x(epochs - 1), height - margin.bottom + 8);
    ctx.fillStyle = "#a0a8b2"; ctx.fillText("epoch", margin.left + plotWidth / 2, height - 10);

    for (const line of group.lines) {
      ctx.beginPath();
      let drawing = false;
      line.values.forEach((value, index) => {
        if (value === null || !Number.isFinite(value)) { drawing = false; return; }
        if (!drawing) { ctx.moveTo(x(index), y(value)); drawing = true; } else { ctx.lineTo(x(index), y(value)); }
      });
      ctx.strokeStyle = colors[line.split] || colors.other;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      const lastIndex = line.values.findLastIndex((value) => value !== null && Number.isFinite(value));
      if (lastIndex >= 0) {
        ctx.beginPath(); ctx.arc(x(lastIndex), y(line.values[lastIndex]), 2.6, 0, Math.PI * 2);
        ctx.fillStyle = colors[line.split] || colors.other; ctx.fill();
      }
    }
  }

  function renderMetadata(items) {
    const section = el("metadata-section");
    const list = el("metadata-list");
    list.replaceChildren();
    if (!items.length) { section.hidden = true; return; }
    section.hidden = false;
    for (const item of items) {
      const details = node("details", "");
      const summary = node("summary", "");
      summary.append(node("span", "", item.label), node("code", "", item.path));
      const pre = node("pre", "", JSON.stringify(item.data, null, 2));
      details.append(summary, pre);
      list.append(details);
    }
  }

  el("refresh-button").addEventListener("click", () => loadTree());
  el("tree-search").addEventListener("input", (event) => {
    state.treeQuery = event.target.value.trim().toLowerCase();
    renderTree();
  });
  el("metric-search").addEventListener("input", (event) => {
    state.query = event.target.value.trim().toLowerCase();
    renderCharts();
  });
  el("source-select").addEventListener("change", (event) => {
    state.source = event.target.value;
    renderCharts();
  });
  document.querySelectorAll("[data-group]").forEach((button) => {
    button.addEventListener("click", () => {
      state.group = button.dataset.group;
      document.querySelectorAll("[data-group]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      renderCharts();
    });
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.charts.forEach(({ canvas, group }) => drawChart(canvas, group)), 100);
  });

  loadTree();
  setInterval(() => loadTree({ quiet: true }), 15000);
})();
