(() => {
  "use strict";

  const storageKey = "mlwiz-dashboard-navigation-v1";
  const themeStorageKey = "mlwiz-dashboard-theme-v1";

  function readStoredState() {
    try {
      return JSON.parse(sessionStorage.getItem(storageKey) || "{}");
    } catch (_error) {
      return {};
    }
  }

  function readStoredTheme() {
    try {
      return localStorage.getItem(themeStorageKey);
    } catch (_error) {
      return null;
    }
  }

  const storedState = readStoredState();
  const storedTheme = readStoredTheme();
  const storedRefreshSeconds = Number(storedState.refreshSeconds);
  const state = {
    tree: null,
    details: null,
    selectedPath: storedState.selectedPath || null,
    openNodes: storedState.openNodes || {},
    treeScrollTop: Number(storedState.treeScrollTop) || 0,
    group: storedState.group || "all",
    source: storedState.source || "all",
    plotMode: storedState.plotMode === "inner-fold-aggregate"
      ? "inner-fold"
      : (storedState.plotMode || "auto"),
    innerFoldAggregate: storedState.innerFoldAggregate
      ?? storedState.plotMode === "inner-fold-aggregate",
    focusedInnerFold: storedState.focusedInnerFold || null,
    focusedRun: storedState.focusedRun || null,
    showAllPlots: Boolean(storedState.showAllPlots),
    query: storedState.query || "",
    scale: storedState.scale || "linear",
    theme: ["dark", "day"].includes(storedTheme)
      ? storedTheme
      : (["dark", "day"].includes(storedState.theme) ? storedState.theme : "dark"),
    refreshSeconds: Number.isFinite(storedRefreshSeconds)
      && storedRefreshSeconds >= 2
      && storedRefreshSeconds <= 3600
      ? Math.round(storedRefreshSeconds)
      : 15,
    experimentFilters: storedState.experimentFilters || {},
    metadataModes: storedState.metadataModes || {},
    metadataScrolls: storedState.metadataScrolls || {},
    overviewExpanded: storedState.overviewExpanded !== false,
    filterData: {},
    filterLoading: {},
    charts: [],
    graphRequestId: 0,
    graphPath: null,
    graphCheckpointChoices: storedState.graphCheckpointChoices || {},
    graphFocusedRuns: storedState.graphFocusedRuns || {},
    graphExpandedNodes: storedState.graphExpandedNodes || {},
    graphZooms: storedState.graphZooms || {},
    graphNodePositions: storedState.graphNodePositions || {},
    graphMode: storedState.graphMode === "operators" ? "operators" : "architecture",
    graphView: storedState.graphView === "leaves" ? "leaves" : "hierarchy",
    graphQuery: "",
    modelGraphData: null,
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
  let refreshTimer = null;
  let metadataScrollTimer = null;
  let graphPointerDrag = null;

  function persistState() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(exportableState()));
    } catch (_error) {
      // The dashboard remains fully usable when browser storage is disabled.
    }
    try {
      localStorage.setItem(themeStorageKey, state.theme);
    } catch (_error) {
      // Keep the theme session-scoped when persistent browser storage is disabled.
    }
  }

  function exportableState() {
    return {
        selectedPath: state.selectedPath,
        openNodes: state.openNodes,
        treeScrollTop: state.treeScrollTop,
        group: state.group,
        source: state.source,
        plotMode: state.plotMode,
        innerFoldAggregate: state.innerFoldAggregate,
        focusedInnerFold: state.focusedInnerFold,
        focusedRun: state.focusedRun,
        showAllPlots: state.showAllPlots,
        query: state.query,
        scale: state.scale,
        theme: state.theme,
        refreshSeconds: state.refreshSeconds,
        experimentFilters: state.experimentFilters,
        metadataModes: state.metadataModes,
        metadataScrolls: state.metadataScrolls,
        overviewExpanded: state.overviewExpanded,
        graphCheckpointChoices: state.graphCheckpointChoices,
        graphFocusedRuns: state.graphFocusedRuns,
        graphExpandedNodes: state.graphExpandedNodes,
        graphZooms: state.graphZooms,
        graphNodePositions: state.graphNodePositions,
        graphMode: state.graphMode,
        graphView: state.graphView,
      };
  }

  async function exportView() {
    const button = el("export-button");
    button.disabled = true;
    button.textContent = "Exporting…";
    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(exportableState()),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `Export failed (${response.status})`);
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "mlwiz-dashboard-view.mlwiz";
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      window.alert(error.message);
    } finally {
      button.disabled = false;
      button.textContent = "⇩ Export all";
    }
  }

  function bindDisclosure(details, key, defaultOpen) {
    details.open = Object.hasOwn(state.openNodes, key)
      ? Boolean(state.openNodes[key])
      : defaultOpen;
    details.addEventListener("toggle", () => {
      state.openNodes[key] = details.open;
      persistState();
    });
  }

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

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function renderCacheStatus(cache) {
    if (!cache) return;
    const usage = el("cache-usage");
    usage.textContent = `Cache · ${formatNumber(cache.used_mb)} / ${formatNumber(cache.max_mb)} MB`;
    usage.title = `${cache.entries} entries · ${cache.hits} hits · ${cache.misses} misses · ${cache.evictions} evictions · ${cache.invalidations} invalidations · ${cache.skipped} oversized/skipped`;
    const input = el("cache-limit");
    if (document.activeElement !== input) input.value = String(cache.max_mb);
  }

  async function loadCacheStatus() {
    try {
      renderCacheStatus(await getJson("/api/cache"));
    } catch (_error) {
      el("cache-usage").textContent = "Cache · unavailable";
    }
  }

  async function applyCacheLimit() {
    const button = el("cache-apply");
    const limit = Number(el("cache-limit").value);
    button.disabled = true;
    button.textContent = "Saving…";
    try {
      renderCacheStatus(await postJson("/api/cache", { max_mb: limit }));
      button.textContent = "Saved";
    } catch (error) {
      button.textContent = "Error";
      button.title = error.message;
    } finally {
      setTimeout(() => {
        button.disabled = false;
        button.textContent = "Apply";
      }, 900);
    }
  }

  async function resetCache() {
    const button = el("cache-reset");
    button.disabled = true;
    button.textContent = "Clearing…";
    try {
      renderCacheStatus(await postJson("/api/cache/reset", {}));
      button.textContent = "Cleared";
    } catch (error) {
      button.textContent = "Error";
      button.title = error.message;
    } finally {
      setTimeout(() => {
        button.disabled = false;
        button.textContent = "Reset";
      }, 900);
    }
  }

  function applyRefreshInterval() {
    const input = el("refresh-interval");
    const button = el("refresh-apply");
    const seconds = Number(input.value);
    if (!Number.isFinite(seconds) || seconds < 2 || seconds > 3600) {
      button.textContent = "2–3600s";
      setTimeout(() => { button.textContent = "Apply"; }, 1200);
      return;
    }
    state.refreshSeconds = Math.round(seconds);
    input.value = String(state.refreshSeconds);
    persistState();
    updateRefreshStatus();
    scheduleRefresh();
    button.textContent = "Saved";
    setTimeout(() => { button.textContent = "Apply"; }, 900);
  }

  function updateRefreshStatus() {
    el("refresh-status").textContent = `Auto-refresh · ${state.refreshSeconds}s`;
  }

  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(async () => {
      await loadTree({ quiet: true });
      scheduleRefresh();
    }, state.refreshSeconds * 1000);
  }

  function applyTheme() {
    document.documentElement.dataset.theme = state.theme;
    const button = el("theme-toggle");
    const dark = state.theme === "dark";
    button.textContent = dark ? "☀ Day" : "◐ Dark";
    button.setAttribute("aria-label", dark ? "Switch to day mode" : "Switch to dark mode");
    button.setAttribute("aria-pressed", String(dark));
  }

  async function loadTree({ quiet = false } = {}) {
    const refresh = el("refresh-button");
    refresh.classList.add("loading");
    try {
      const tree = await getJson("/api/tree");
      state.tree = tree;
      await Promise.all(
        tree.experiments
          .filter((experiment) => state.experimentFilters[experiment.path]?.enabled)
          .map((experiment) => loadExperimentFilter(experiment.path, false))
      );
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

  function renderTree() {
    const scrollTop = treeElement.scrollTop || state.treeScrollTop;
    treeElement.replaceChildren();
    if (!state.tree || state.tree.experiments.length === 0) {
      const empty = node("div", "tree-empty");
      empty.append(node("strong", "", "No MLWiz experiments found"));
      empty.append(document.createTextNode("Point --logdir to a folder containing MODEL_ASSESSMENT results."));
      treeElement.append(empty);
      return;
    }

    for (const experiment of state.tree.experiments) {
      const expDetails = node("details", "tree-group");
      bindDisclosure(expDetails, `experiment:${experiment.path}`, true);
      const summary = node("summary", "experiment-summary");
      summary.append(
        node("span", "chevron", "›"),
        node("span", "experiment-icon", "MW"),
        node("span", "", experiment.name),
        node("span", "tree-count", `${experiment.run_count} runs`)
      );
      expDetails.append(summary);
      expDetails.append(renderExperimentFilter(experiment));

      for (const fold of experiment.outer_folds) {
        const foldElement = renderFold(fold, experiment.path);
        if (foldElement) expDetails.append(foldElement);
      }
      treeElement.append(expDetails);
    }
    treeElement.scrollTop = scrollTop;
  }

  function renderFold(fold, experimentPath) {
    const visibleConfigs = fold.model_selection.filter((config) =>
      configurationPassesFilter(experimentPath, config.path)
    );
    const filtering = activeFilterClauses(experimentPath).length > 0;
    const visibleFinalRuns = filtering ? [] : fold.final_runs;
    if (!visibleConfigs.length && !visibleFinalRuns.length) return null;

    const foldDetails = node("details", "fold-group");
    bindDisclosure(foldDetails, `fold:${fold.path}`, true);
    const summary = node("summary", "fold-summary");
    summary.append(node("span", "chevron", "›"), node("span", "", `Outer fold ${fold.number}`));
    foldDetails.append(summary);

    if (visibleConfigs.length) {
      foldDetails.append(node("div", "tree-section-label", "Model selection"));
      for (const config of visibleConfigs) {
        const configDetails = node("details", "config-group");
        const disclosureKey = `configuration:${config.path}`;
        const containsSelection = state.selectedPath === config.path
          || state.selectedPath?.startsWith(`${config.path}/`);
        bindDisclosure(configDetails, disclosureKey, containsSelection);
        const configSummary = node("summary", "config-summary");
        if (config.path === state.selectedPath) configSummary.classList.add("active");
        configSummary.append(node("span", "chevron", "›"));
        const configLink = node("span", "", `Configuration ${config.number}`);
        configSummary.append(configLink);
        if (config.is_winner) configSummary.append(node("span", "winner", "winner"));
        configSummary.addEventListener("click", (event) => {
          event.preventDefault();
          configDetails.open = !configDetails.open;
          state.openNodes[disclosureKey] = configDetails.open;
          persistState();
          loadDetails(config.path);
        });
        configDetails.append(configSummary);
        for (const inner of config.inner_folds) {
          configDetails.append(node("div", "inner-label", `Inner fold ${inner.number}`));
          for (const run of inner.runs) configDetails.append(runButton(run, `Run ${run.number}`));
        }
        foldDetails.append(configDetails);
      }
    }

    if (visibleFinalRuns.length) {
      foldDetails.append(node("div", "tree-section-label", "Final runs"));
      for (const run of visibleFinalRuns) {
        foldDetails.append(runButton(run, `Final run ${run.number}`));
      }
    }
    return foldDetails;
  }

  async function loadExperimentFilter(experimentPath, rerender = true) {
    if (state.filterLoading[experimentPath]) return;
    state.filterLoading[experimentPath] = true;
    if (rerender) renderTree();
    try {
      const data = await getJson(`/api/experiment-filter?path=${encodeURIComponent(experimentPath)}`);
      state.filterData[experimentPath] = data;
      renderCacheStatus(data.cache);
      const definition = state.experimentFilters[experimentPath];
      if (definition && (!definition.clauses || !definition.clauses.length)) {
        definition.clauses = [newFilterClause(data)];
        definition.logic = definition.logic || "AND";
        persistState();
      } else if (definition) {
        const metricIds = new Set(data.metrics.map((metric) => metric.id));
        for (const clause of definition.clauses) {
          if (!metricIds.has(clause.metric)) clause.metric = data.default_metric;
          if (!data.splits.includes(clause.split)) clause.split = data.default_split;
        }
        persistState();
      }
    } catch (error) {
      state.filterData[experimentPath] = { error: error.message };
    } finally {
      state.filterLoading[experimentPath] = false;
      if (rerender) renderTree();
    }
  }

  function renderExperimentFilter(experiment) {
    const panel = node("div", "experiment-filter");
    const definition = state.experimentFilters[experiment.path];
    if (!definition?.enabled) {
      const launch = node("button", "filter-launch", "⌕ Filter configurations");
      launch.type = "button";
      launch.addEventListener("click", () => {
        state.experimentFilters[experiment.path] = {
          enabled: true,
          logic: "AND",
          clauses: [],
        };
        persistState();
        loadExperimentFilter(experiment.path);
      });
      panel.append(launch);
      return panel;
    }

    const data = state.filterData[experiment.path];
    if (state.filterLoading[experiment.path] || !data) {
      panel.append(node("span", "filter-loading", "Loading experiment metrics…"));
      return panel;
    }
    if (data.error) {
      const retry = node("button", "filter-launch filter-error", `Retry filter · ${data.error}`);
      retry.type = "button";
      retry.addEventListener("click", () => loadExperimentFilter(experiment.path));
      panel.append(retry);
      return panel;
    }

    const header = node("div", "filter-head");
    header.append(
      node("span", "filter-status", activeFilterClauses(experiment.path).length ? "Filtering configurations" : "Configuration filters"),
      node("span", "filter-source", data.complete ? "aggregated result" : "last epoch")
    );
    panel.append(header);

    const clauses = definition.clauses || [];
    if (clauses.length > 1) {
      const combine = node("label", "filter-combine");
      combine.append(node("span", "", "Match"));
      const logic = node("select", "filter-logic");
      for (const [value, label] of [["AND", "all conditions (AND)"], ["OR", "any condition (OR)"]]) {
        const option = node("option", "", label);
        option.value = value;
        logic.append(option);
      }
      logic.value = definition.logic || "AND";
      logic.setAttribute("aria-label", "Filter combination operator");
      logic.addEventListener("change", (event) => {
        definition.logic = event.target.value;
        persistState();
        renderTree();
      });
      combine.append(logic);
      panel.append(combine);
    }
    clauses.forEach((clause, index) => {
      panel.append(renderFilterClause(experiment.path, clause, index, data.metrics));
    });

    const actions = node("div", "filter-actions");
    const add = node("button", "", "+ Add condition");
    add.type = "button";
    add.addEventListener("click", () => {
      definition.clauses.push(newFilterClause(data));
      persistState();
      renderTree();
    });
    const clear = node("button", "", "Clear");
    clear.type = "button";
    clear.addEventListener("click", () => {
      definition.clauses = [newFilterClause(data)];
      persistState();
      renderTree();
    });
    actions.append(add, clear);
    panel.append(actions);
    return panel;
  }

  function renderFilterClause(experimentPath, clause, index, metrics) {
    const row = node("div", "filter-clause");
    const split = node("select", "filter-split");
    split.setAttribute("aria-label", "Training or validation metric");
    for (const value of state.filterData[experimentPath].splits) {
      const option = node("option", "", value[0].toUpperCase() + value.slice(1));
      option.value = value;
      split.append(option);
    }
    split.value = clause.split;
    split.addEventListener("change", (event) => updateFilterClause(experimentPath, index, "split", event.target.value));

    const metric = node("select", "filter-metric");
    metric.setAttribute("aria-label", "Metric to filter");
    for (const [kind, label] of [["score", "Scores"], ["loss", "Losses"]]) {
      const optionGroup = node("optgroup", "");
      optionGroup.label = label;
      for (const descriptor of metrics.filter((item) => item.kind === kind)) {
        const option = node("option", "", descriptor.label);
        option.value = descriptor.id;
        optionGroup.append(option);
      }
      if (optionGroup.children.length) metric.append(optionGroup);
    }
    metric.value = clause.metric;
    metric.addEventListener("change", (event) => updateFilterClause(experimentPath, index, "metric", event.target.value));

    const operator = node("select", "filter-operator");
    operator.setAttribute("aria-label", "Metric comparison");
    for (const [value, label] of [["gte", "≥"], ["lte", "≤"]]) {
      const option = node("option", "", label);
      option.value = value;
      operator.append(option);
    }
    operator.value = clause.operator;
    operator.addEventListener("change", (event) => updateFilterClause(experimentPath, index, "operator", event.target.value));

    const threshold = node("input", "filter-value");
    threshold.type = "number";
    threshold.step = "any";
    threshold.placeholder = "value";
    threshold.value = clause.value;
    threshold.setAttribute("aria-label", "Filter threshold");
    threshold.addEventListener("change", (event) => updateFilterClause(experimentPath, index, "value", event.target.value));

    const remove = node("button", "filter-remove", "×");
    remove.type = "button";
    remove.title = "Remove condition";
    remove.addEventListener("click", () => {
      const definition = state.experimentFilters[experimentPath];
      definition.clauses.splice(index, 1);
      if (!definition.clauses.length) {
        definition.clauses.push(newFilterClause(state.filterData[experimentPath]));
      }
      persistState();
      renderTree();
    });
    row.append(split, metric, operator, threshold, remove);
    return row;
  }

  function updateFilterClause(experimentPath, index, key, value) {
    const clause = state.experimentFilters[experimentPath].clauses[index];
    clause[key] = value;
    if (key === "metric") {
      const descriptor = state.filterData[experimentPath].metrics.find((metric) => metric.id === value);
      clause.operator = descriptor?.kind === "loss" ? "lte" : "gte";
    }
    persistState();
    renderTree();
  }

  function newFilterClause(data) {
    const descriptor = data.metrics.find((metric) => metric.id === data.default_metric);
    return {
      metric: data.default_metric,
      split: data.default_split,
      operator: descriptor?.kind === "loss" ? "lte" : "gte",
      value: "",
    };
  }

  function activeFilterClauses(experimentPath) {
    const definition = state.experimentFilters[experimentPath];
    if (!definition?.enabled) return [];
    return (definition.clauses || []).filter((clause) =>
      clause.metric && clause.value !== "" && Number.isFinite(Number(clause.value))
    );
  }

  function configurationPassesFilter(experimentPath, configPath) {
    const clauses = activeFilterClauses(experimentPath);
    const data = state.filterData[experimentPath];
    if (!clauses.length || !data?.configurations) return true;
    const values = data.configurations[configPath]?.values || {};
    const matches = clauses.map((clause) => {
      const metricValue = values[`${clause.split}:${clause.metric}`];
      if (!Number.isFinite(metricValue)) return false;
      return clause.operator === "lte"
        ? metricValue <= Number(clause.value)
        : metricValue >= Number(clause.value);
    });
    return state.experimentFilters[experimentPath].logic === "OR"
      ? matches.some(Boolean)
      : matches.every(Boolean);
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
    const selectionChanged = state.selectedPath !== path;
    state.selectedPath = path;
    if (selectionChanged) {
      state.source = "all";
      state.plotMode = "auto";
      state.focusedInnerFold = null;
      state.focusedRun = null;
      state.showAllPlots = false;
    }
    persistState();
    detailsView.hidden = false;
    welcome.hidden = true;
    renderTree();
    if (!quiet) {
      el("selection-kind").textContent = "Loading metrics";
      el("selection-name").textContent = path.split("/").pop();
      el("selection-path").textContent = path;
      el("chart-grid").replaceChildren(chartMessage("Reading metrics_data.torch…"));
    }
    try {
      const aggregateFinalRuns = state.plotMode === "final-aggregate"
        ? "&aggregate_final_runs=1"
        : "";
      state.details = await getJson(`/api/details?path=${encodeURIComponent(path)}${aggregateFinalRuns}`);
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
    renderCacheStatus(data.cache);
    renderExperimentOverview(data.overview);
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
    renderPlotMode();
    renderInnerFoldAggregation();
    renderSourceFilter(sources);
    renderPlotNavigator();

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
    prepareModelGraph();
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

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
    const total = Math.max(0, Math.round(Number(seconds)));
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (days) return `${days}d ${hours}h`;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  }

  function renderExperimentOverview(overview) {
    const host = el("experiment-overview");
    host.replaceChildren();
    if (!overview) return;

    const box = node("details", "experiment-overview-box");
    box.open = state.overviewExpanded;
    const summary = node("summary", "overview-summary");
    const identity = node("div", "overview-identity");
    const statusDot = node("span", `overview-status-dot ${overview.state}`);
    const nameBlock = node("div", "");
    nameBlock.append(
      node("span", "overview-kicker", "Main experiment"),
      node("strong", "", overview.name)
    );
    identity.append(statusDot, nameBlock);
    const summaryMeta = node("div", "overview-summary-meta");
    const runSummary = overview.runs.total
      ? `${overview.runs.completed} of ${overview.runs.total} runs complete`
      : "No runs discovered";
    summaryMeta.append(
      node("span", `overview-state ${overview.state}`, overview.state),
      node("span", "overview-run-summary", runSummary),
      node("span", "overview-chevron", "⌄")
    );
    summary.append(identity, summaryMeta);
    box.append(summary);

    const body = node("div", "overview-body");
    if (overview.runs.total) {
      const progress = node("div", "overview-progress");
      const progressFill = node("span", "");
      progressFill.style.width = `${Math.min(100, (overview.runs.completed / overview.runs.total) * 100)}%`;
      progress.append(progressFill);
      body.append(progress);
    }

    const activity = node("div", "overview-activity");
    for (const [label, value, className] of [
      ["Completed", overview.runs.completed, "completed"],
      ["Running", overview.runs.running, "running"],
      ["Queued", overview.runs.queued, "queued"],
      ["Failed", overview.runs.failed, "failed"],
    ]) {
      if (value || label === "Completed") {
        const chip = node("span", `overview-chip ${className}`);
        chip.append(node("strong", "", String(value)), document.createTextNode(` ${label}`));
        activity.append(chip);
      }
    }
    const configText = `${overview.configurations.completed} / ${overview.configurations.total} configurations aggregated`;
    activity.append(node("span", "overview-config-count", configText));
    body.append(activity);

    const timing = overview.timing;
    const timingGrid = node("div", "overview-timing-grid");
    for (const [label, value] of [
      ["Recorded compute", timing.recorded_total_seconds],
      ["Average run", timing.average_run_seconds],
      ["Median run", timing.median_run_seconds],
      [overview.state === "completed" ? "Slowest run" : "Remaining compute", overview.state === "completed" ? timing.slowest_run_seconds : timing.estimated_remaining_compute_seconds],
    ]) {
      const card = node("div", "overview-time-card");
      card.append(node("span", "", label), node("strong", "", formatDuration(value)));
      timingGrid.append(card);
    }
    body.append(timingGrid);

    const note = overview.state === "completed"
      ? `${timing.timed_runs} completed runs include timing markers.`
      : "Remaining compute is estimated as average completed-run time × unfinished runs; parallel execution can reduce wall time.";
    body.append(node("p", "overview-note", note));
    box.append(body);
    box.addEventListener("toggle", () => {
      state.overviewExpanded = box.open;
      persistState();
    });
    host.append(box);
  }

  function plotModeOptions() {
    const scope = state.details?.selection.plot_scope;
    if (scope === "model_selection_configuration") {
      return [
        ["individual", "Individual runs"],
        ["inner-fold", "Group by inner fold"],
      ];
    }
    if (scope === "final_runs") {
      return [
        ["final-selected", "Selected final run"],
        ["final-aggregate", "All final runs mean ± std"],
      ];
    }
    return [];
  }

  function resolvedPlotMode() {
    const options = plotModeOptions();
    const allowed = options.map(([value]) => value);
    return allowed.includes(state.plotMode) ? state.plotMode : (allowed[0] || "individual");
  }

  function renderPlotMode() {
    const field = el("plot-mode-field");
    const select = el("plot-mode-select");
    const options = plotModeOptions();
    select.replaceChildren();
    if (!options.length) {
      field.hidden = true;
      return;
    }
    field.hidden = false;
    const mode = resolvedPlotMode();
    if (state.plotMode !== mode) {
      state.plotMode = mode;
      persistState();
    }
    for (const [value, label] of options) {
      const option = node("option", "", label);
      option.value = value;
      select.append(option);
    }
    select.value = mode;
  }

  function renderInnerFoldAggregation() {
    const field = el("inner-fold-aggregate-field");
    const checkbox = el("inner-fold-aggregate");
    const visible = state.details?.selection.plot_scope === "model_selection_configuration"
      && resolvedPlotMode() === "inner-fold";
    field.hidden = !visible;
    checkbox.checked = Boolean(state.innerFoldAggregate);
  }

  function renderSourceFilter(sources) {
    const field = el("source-field");
    const select = el("source-select");
    select.replaceChildren();
    if (state.details?.selection.plot_scope === "model_selection_configuration") {
      field.hidden = true;
      return;
    }
    if (state.source !== "all" && !sources.includes(state.source)) {
      state.source = "all";
      persistState();
    }
    if (resolvedPlotMode() !== "individual" || sources.length <= 1) {
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

  function naturalSort(values) {
    return [...values].sort((left, right) => left.localeCompare(
      right,
      undefined,
      { numeric: true, sensitivity: "base" },
    ));
  }

  function configurationRunSources() {
    return naturalSort(new Set(
      state.details.series
        .map((series) => series.source)
        .filter((source) => innerFoldName(source) !== "Ungrouped"),
    ));
  }

  function setSelectOptions(select, values, selected, labeler = (value) => value) {
    select.replaceChildren();
    for (const value of values) {
      const option = node("option", "", labeler(value));
      option.value = value;
      select.append(option);
    }
    select.value = selected;
  }

  function renderPlotNavigator() {
    const navigator = el("plot-navigator");
    const isConfiguration = state.details?.selection.plot_scope === "model_selection_configuration";
    if (!isConfiguration) {
      navigator.hidden = true;
      return;
    }

    const sources = configurationRunSources();
    const folds = naturalSort(new Set(sources.map(innerFoldName)));
    if (!folds.length) {
      navigator.hidden = true;
      return;
    }
    navigator.hidden = false;

    let changed = false;
    if (!folds.includes(state.focusedInnerFold)) {
      state.focusedInnerFold = folds[0];
      changed = true;
    }
    const foldRuns = sources.filter(
      (source) => innerFoldName(source) === state.focusedInnerFold,
    );
    if (!foldRuns.includes(state.focusedRun)) {
      state.focusedRun = foldRuns[0] || null;
      changed = true;
    }
    if (changed) persistState();

    setSelectOptions(el("fold-select"), folds, state.focusedInnerFold);
    setSelectOptions(el("run-select"), foldRuns, state.focusedRun, shortRunName);
    const individual = resolvedPlotMode() === "individual";
    el("run-navigator").hidden = !individual;
    el("show-all-plots").checked = state.showAllPlots;
    for (const control of [
      el("fold-select"), el("fold-previous"), el("fold-next"),
      el("run-select"), el("run-previous"), el("run-next"),
    ]) control.disabled = state.showAllPlots;

    el("plot-navigator-status").textContent = state.showAllPlots
      ? (individual ? "All folds and runs" : "All inner folds")
      : (individual
        ? `${state.focusedInnerFold} · ${shortRunName(state.focusedRun)}`
        : state.focusedInnerFold);
  }

  function moveNavigatorSelection(selectId, offset) {
    const select = el(selectId);
    if (select.disabled || select.options.length < 2) return;
    const nextIndex = (select.selectedIndex + offset + select.options.length)
      % select.options.length;
    select.selectedIndex = nextIndex;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function observeStickyPlotNavigator() {
    const navigator = el("plot-navigator");
    const sentinel = el("plot-navigator-sentinel");
    const observer = new IntersectionObserver(([entry]) => {
      const passedStickyEdge = entry.boundingClientRect.top <= 10;
      navigator.classList.toggle("is-stuck", !entry.isIntersecting && passedStickyEdge);
    }, {
      threshold: 0,
      rootMargin: "-10px 0px 0px",
    });
    observer.observe(sentinel);
  }

  function splitMetric(name) {
    for (const split of ["training", "validation", "test"]) {
      if (name.startsWith(`${split}_`)) return { split, metric: name.slice(split.length + 1) };
    }
    return { split: "other", metric: name };
  }

  function innerFoldName(source) {
    const match = source.match(/(?:^|\/)(INNER_FOLD_\d+)(?:\/|$)/);
    return match ? match[1] : "Ungrouped";
  }

  function shortRunName(source) {
    return source.split("/").pop().replaceAll("_", " ");
  }

  function runDashPattern(source) {
    const match = source.match(/run_?(\d+)$/i);
    const index = match ? Number(match[1]) - 1 : 0;
    return [[], [7, 3], [2, 3], [9, 3, 2, 3]][Math.abs(index) % 4];
  }

  function aggregateMetricLines(lines) {
    const epochs = Math.max(...lines.map((line) => line.values.length), 0);
    const values = [];
    const lower = [];
    const upper = [];
    for (let index = 0; index < epochs; index += 1) {
      const samples = lines
        .map((line) => line.values[index])
        .filter((value) => value !== null && Number.isFinite(value));
      if (!samples.length) {
        values.push(null); lower.push(null); upper.push(null);
        continue;
      }
      const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
      const variance = samples.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / samples.length;
      const std = Math.sqrt(variance);
      values.push(mean);
      lower.push(mean - std);
      upper.push(mean + std);
    }
    return { values, band: { lower, upper }, sampleCount: lines.length };
  }

  function groupedSeries() {
    const mode = resolvedPlotMode();
    const isConfiguration = state.details.selection.plot_scope === "model_selection_configuration";
    const prepared = [];
    for (const series of state.details.series) {
      if (state.group !== "all" && series.group !== state.group) continue;
      if (!isConfiguration && mode === "individual" && state.source !== "all" && series.source !== state.source) continue;
      if (mode === "final-selected" && series.source !== state.details.selection.selected_source) continue;
      const { split, metric } = splitMetric(series.name);
      const innerFold = innerFoldName(series.source);
      if (isConfiguration && !state.showAllPlots) {
        if (mode === "inner-fold" && innerFold !== state.focusedInnerFold) continue;
        if (mode === "individual" && series.source !== state.focusedRun) continue;
      }
      if (state.query && !`${series.name} ${series.source} ${series.group}`.toLowerCase().includes(state.query)) continue;
      prepared.push({ ...series, split, metric, innerFold });
    }

    const groups = new Map();
    const aggregateInnerFolds = mode === "inner-fold" && state.innerFoldAggregate;
    if (aggregateInnerFolds || mode === "final-aggregate") {
      const buckets = new Map();
      for (const series of prepared) {
        const scope = aggregateInnerFolds ? series.innerFold : "All final runs";
        const key = `${scope}\u0000${series.group}\u0000${series.metric}\u0000${series.split}`;
        if (!buckets.has(key)) buckets.set(key, { scope, group: series.group, metric: series.metric, split: series.split, lines: [] });
        buckets.get(key).lines.push(series);
      }
      for (const bucket of buckets.values()) {
        const key = `${bucket.scope}\u0000${bucket.group}\u0000${bucket.metric}`;
        if (!groups.has(key)) {
          const suffix = aggregateInnerFolds ? "mean ± std across runs" : "mean ± std";
          groups.set(key, { source: `${bucket.scope} · ${suffix}`, group: bucket.group, metric: bucket.metric, lines: [] });
        }
        const aggregate = aggregateMetricLines(bucket.lines);
        groups.get(key).lines.push({
          id: `${key}\u0000${bucket.split}`,
          split: bucket.split,
          label: `${bucket.split} mean ± std (n=${aggregate.sampleCount})`,
          ...aggregate,
        });
      }
    } else {
      for (const series of prepared) {
        const scope = mode === "inner-fold" ? series.innerFold : series.source;
        const key = `${scope}\u0000${series.group}\u0000${series.metric}`;
        if (!groups.has(key)) groups.set(key, { source: scope, group: series.group, metric: series.metric, lines: [] });
        groups.get(key).lines.push({
          id: `${series.source}\u0000${series.name}`,
          split: series.split,
          label: mode === "inner-fold" ? `${series.split} · ${shortRunName(series.source)}` : series.split,
          name: series.name,
          values: series.values,
          dash: mode === "inner-fold" ? runDashPattern(series.source) : [],
        });
      }
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
      const headMeta = node("div", "chart-head-meta");
      const epochLabel = node("span", "chart-epoch", "Latest");
      headMeta.append(epochLabel, node("span", "chart-type", group.group));
      head.append(title, headMeta);
      const wrap = node("div", "chart-wrap");
      const canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", `${group.metric} over epochs`);
      wrap.append(canvas);
      const legend = node("div", "chart-legend");
      const legendValues = new Map();
      for (const line of group.lines) {
        const item = node("span", "legend-item");
        const swatch = node("span", "legend-swatch");
        if (line.band) swatch.classList.add("band");
        swatch.style.background = colors[line.split] || colors.other;
        const value = node("span", "legend-value", formatLineReadout(line));
        legendValues.set(line.id, value);
        item.append(swatch, document.createTextNode(`${line.label} `), value);
        legend.append(item);
      }
      card.append(head, wrap, legend);
      grid.append(card);
      const chart = {
        canvas,
        group,
        legendValues,
        epochLabel,
        hoverIndex: null,
      };
      canvas.addEventListener("pointermove", (event) => updateChartHover(chart, event));
      canvas.addEventListener("pointerleave", () => {
        chart.hoverIndex = null;
        updateChartReadout(chart);
        drawChart(chart);
      });
      state.charts.push(chart);
      updateChartReadout(chart);
      drawChart(chart);
    }
  }

  function renderChartsPreservingScroll() {
    const scrollX = window.scrollX;
    const scrollY = window.scrollY;
    renderCharts();
    window.scrollTo(scrollX, scrollY);
    requestAnimationFrame(() => window.scrollTo(scrollX, scrollY));
  }

  function graphRunKey(selection) {
    return `${selection.path}:${state.focusedInnerFold}`;
  }

  function graphRunCandidates(selection) {
    return configurationRunSources().filter(
      (source) => innerFoldName(source) === state.focusedInnerFold,
    );
  }

  function renderModelGraphRunSelector() {
    const control = el("model-graph-run-control");
    const select = el("model-graph-run-select");
    const selection = state.details?.selection;
    const groupedConfiguration = selection?.plot_scope === "model_selection_configuration"
      && resolvedPlotMode() === "inner-fold";
    if (!groupedConfiguration) {
      control.hidden = true;
      return;
    }
    const candidates = graphRunCandidates(selection);
    if (!candidates.length) {
      control.hidden = true;
      return;
    }
    const key = graphRunKey(selection);
    let selected = state.graphFocusedRuns[key];
    if (!candidates.includes(selected)) {
      selected = candidates.includes(state.focusedRun) ? state.focusedRun : candidates[0];
      state.graphFocusedRuns[key] = selected;
      persistState();
    }
    setSelectOptions(select, candidates, selected, shortRunName);
    control.hidden = false;
  }

  function focusedModelGraphPath() {
    const selection = state.details?.selection;
    if (!selection) return null;
    if (selection.plot_scope === "model_selection_configuration") {
      const graphRun = resolvedPlotMode() === "inner-fold"
        ? state.graphFocusedRuns[graphRunKey(selection)]
        : state.focusedRun;
      return graphRun ? `${selection.path}/${graphRun}` : null;
    }
    if (["single_run", "final_runs"].includes(selection.plot_scope)) {
      return selection.path;
    }
    return null;
  }

  function prepareModelGraph() {
    const section = el("model-graph-section");
    renderModelGraphRunSelector();
    const path = focusedModelGraphPath();
    section.hidden = !path;
    if (!path) return;
    el("model-graph-run").textContent = path.split("/").slice(-2).join(" / ").replaceAll("_", " ");
    if (section.open) loadModelGraph(path);
  }

  async function loadModelGraph(path) {
    const requestId = ++state.graphRequestId;
    const pathChanged = state.graphPath !== path;
    state.graphPath = path;
    const status = el("model-graph-status");
    const message = el("model-graph-message");
    let checkpointChoice = state.graphCheckpointChoices[path] || "auto";
    el("model-graph-mode-select").disabled = true;
    el("model-graph-checkpoint-select").disabled = true;
    if (pathChanged) {
      el("model-graph-canvas").replaceChildren();
      el("model-graph-stats").replaceChildren();
      el("model-node-details").textContent = state.graphMode === "operators"
        ? "Select an operator to inspect its exported tensor metadata."
        : "Select a module to inspect its checkpoint tensors.";
      message.hidden = true;
      status.textContent = "Loading checkpoint graph on CPU…";
    } else {
      status.textContent = "Checking the active checkpoint…";
      message.hidden = true;
    }
    try {
      const info = await getJson(`/api/model-graph-info?path=${encodeURIComponent(path)}`);
      if (requestId !== state.graphRequestId || path !== focusedModelGraphPath()) return;
      renderGraphModeSelect(info);
      if (checkpointChoice !== "auto" && !info.checkpoint.available.includes(checkpointChoice)) {
        checkpointChoice = "auto";
        state.graphCheckpointChoices[path] = "auto";
        persistState();
      }
      renderGraphCheckpointSelect(info);
      const resolvedKind = checkpointChoice === "auto"
        ? info.checkpoint.kind
        : checkpointChoice;
      if (!resolvedKind) {
        throw new Error("No best or last checkpoint is available for this run.");
      }
      if (!info.checkpoint.loadable.includes(resolvedKind)) {
        const size = info.checkpoint.sizes_mb[resolvedKind];
        status.textContent = "Model graph not loaded";
        message.hidden = false;
        message.className = "model-graph-message error";
        message.textContent = `${resolvedKind[0].toUpperCase()}${resolvedKind.slice(1)} checkpoint is ${formatNumber(size)} MB, exceeding the ${formatNumber(info.cache_max_mb)} MB cache limit.`;
        return;
      }
      const graph = await getJson(`/api/model-graph?path=${encodeURIComponent(path)}&checkpoint=${encodeURIComponent(checkpointChoice)}&mode=${encodeURIComponent(state.graphMode)}`);
      if (requestId !== state.graphRequestId || path !== focusedModelGraphPath()) return;
      renderModelGraph(graph);
    } catch (error) {
      if (requestId !== state.graphRequestId) return;
      status.textContent = "Model graph unavailable";
      message.hidden = false;
      message.className = "model-graph-message error";
      message.textContent = error.message;
      el("model-graph-mode-select").disabled = false;
      el("model-graph-checkpoint-select").disabled = false;
    }
  }

  function graphSvgElement(tag, attributes = {}, text = null) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== null) element.textContent = text;
    return element;
  }

  function renderModelGraph(graph) {
    renderCacheStatus(graph.cache);
    const previousMode = state.modelGraphData?.graph_mode;
    state.modelGraphData = graph;
    if (previousMode && previousMode !== graph.graph_mode) {
      el("model-node-details").textContent = graph.graph_mode === "operators"
        ? "Select an operator to inspect its exported tensor metadata."
        : "Select a module to inspect its checkpoint tensors.";
    }
    const epoch = graph.epoch ? ` · epoch ${graph.epoch}` : "";
    const modeLabel = graph.graph_mode === "operators" ? "Operators" : "Architecture";
    el("model-graph-status").textContent = `${graph.run_state} · ${graph.checkpoint.kind} checkpoint${epoch} · ${modeLabel}`;
    const message = el("model-graph-message");
    if (graph.warning) {
      message.hidden = false;
      message.className = "model-graph-message";
      message.textContent = graph.warning;
    } else {
      message.hidden = true;
    }
    const stats = el("model-graph-stats");
    stats.replaceChildren();
    const nodeLabel = graph.graph_mode === "operators"
      ? `${formatNumber(graph.summary.modules)} modules · ${formatNumber(graph.summary.operators)} operators`
      : `${graph.summary.visible_nodes} module${graph.summary.visible_nodes === 1 ? "" : "s"}`;
    for (const text of [
      graph.summary.model_class,
      `${formatNumber(graph.summary.parameters)} parameters`,
      nodeLabel,
      graph.checkpoint.cache_hit ? "Graph cache hit" : "Loaded from checkpoint",
    ]) stats.append(node("span", "model-graph-stat", text));
    if (graph.summary.truncated) {
      stats.append(node("span", "model-graph-stat", `Showing first ${graph.summary.visible_nodes} of ${graph.summary.nodes}`));
    }
    syncGraphExplorerControls();
    renderModelGraphCanvas();
  }

  function syncGraphExplorerControls() {
    const architecture = state.modelGraphData?.graph_mode !== "operators";
    const leaves = state.graphView === "leaves";
    const button = el("graph-view-toggle");
    button.classList.toggle("active", leaves);
    button.setAttribute("aria-pressed", String(leaves));
    button.textContent = leaves ? "Show hierarchy" : "Flatten to leaves";
    el("graph-search").value = state.graphQuery;
    el("graph-search").placeholder = architecture ? "Find module…" : "Find module or operator…";
    button.hidden = !architecture;
    el("graph-expand-all").disabled = architecture && leaves;
    el("graph-collapse-all").disabled = architecture && leaves;
    el("graph-zoom-controls").hidden = architecture;
    syncGraphZoomControls();
  }

  function graphExpansionKey(graph) {
    return `${graph.run}:${graph.checkpoint.kind}:${graph.graph_mode}`;
  }

  function graphZoomKey(graph) {
    return `${graphExpansionKey(graph)}:zoom`;
  }

  function graphPositionKey(graph) {
    return `${graphExpansionKey(graph)}:positions`;
  }

  function applyOperatorPositionOverrides(graph, items, layout) {
    const stored = state.graphNodePositions[graphPositionKey(graph)] || {};
    for (const item of items) {
      const position = stored[item.id];
      if (Number.isFinite(position?.x) && Number.isFinite(position?.y)) {
        layout.positions.set(item.id, {
          x: Math.max(8, position.x),
          y: Math.max(34, position.y),
        });
      }
    }
    const points = [...layout.positions.values()];
    return {
      ...layout,
      width: Math.max(layout.width, ...points.map((point) => point.x + 222)),
      height: Math.max(layout.height, ...points.map((point) => point.y + 90)),
    };
  }

  function currentGraphZoom(graph = state.modelGraphData) {
    if (!graph) return 1;
    const zoom = Number(state.graphZooms[graphZoomKey(graph)]);
    return Number.isFinite(zoom) ? Math.max(0.35, Math.min(2.5, zoom)) : 1;
  }

  function syncGraphZoomControls() {
    const zoom = currentGraphZoom();
    el("graph-zoom-reset").textContent = `${Math.round(zoom * 100)}%`;
    el("graph-zoom-out").disabled = zoom <= 0.35;
    el("graph-zoom-in").disabled = zoom >= 2.5;
  }

  function setGraphZoom(nextZoom, anchor = null) {
    const graph = state.modelGraphData;
    if (!graph || graph.graph_mode !== "operators") return;
    const host = el("model-graph-canvas");
    const previousZoom = currentGraphZoom(graph);
    const anchorX = anchor?.x ?? (host.clientWidth / 2);
    const anchorY = anchor?.y ?? (host.clientHeight / 2);
    const graphX = (host.scrollLeft + anchorX) / previousZoom;
    const graphY = (host.scrollTop + anchorY) / previousZoom;
    const zoom = Math.max(0.35, Math.min(2.5, nextZoom));
    state.graphZooms[graphZoomKey(graph)] = zoom;
    persistState();
    renderModelGraphCanvas();
    requestAnimationFrame(() => {
      host.scrollLeft = Math.max(0, (graphX * zoom) - anchorX);
      host.scrollTop = Math.max(0, (graphY * zoom) - anchorY);
    });
  }

  function buildGraphExplorerModel(graph) {
    const nodes = new Map(graph.nodes.map((item) => [item.id, { ...item }]));
    const children = new Map(graph.nodes.map((item) => [item.id, []]));
    const parents = new Map();
    for (const edge of graph.edges) {
      if (!nodes.has(edge.source) || !nodes.has(edge.target)) continue;
      children.get(edge.source).push(edge.target);
      parents.set(edge.target, edge.source);
    }
    const roots = graph.nodes
      .map((item) => item.id)
      .filter((id) => !parents.has(id));
    const totals = new Map();
    const calculateTotal = (id, active = new Set()) => {
      if (totals.has(id)) return totals.get(id);
      if (active.has(id)) return 0;
      const nextActive = new Set(active).add(id);
      const total = (nodes.get(id)?.parameters || 0)
        + (children.get(id) || []).reduce(
          (sum, child) => sum + calculateTotal(child, nextActive),
          0,
        );
      totals.set(id, total);
      return total;
    };
    for (const id of nodes.keys()) calculateTotal(id);
    const modelTotal = Math.max(Number(graph.summary.parameters) || 0, 1);
    for (const item of nodes.values()) {
      item.blockParameters = totals.get(item.id) || 0;
      item.parameterShare = Math.min(1, item.blockParameters / modelTotal);
    }
    return { nodes, children, parents, roots };
  }

  function graphParameterColor(share) {
    const amount = Math.max(0, Math.min(1, Number(share) || 0)) ** 0.38;
    const low = state.theme === "dark" ? [24, 44, 55] : [236, 246, 242];
    const high = state.theme === "dark" ? [145, 53, 39] : [226, 82, 55];
    const channels = low.map((value, index) => Math.round(
      value + ((high[index] - value) * amount),
    ));
    return {
      color: `rgb(${channels.join(",")})`,
      dense: amount > 0.58,
    };
  }

  function visibleGraphItems(graph, explorer) {
    const expanded = new Set(
      state.graphExpandedNodes[graphExpansionKey(graph)] || [],
    );
    const query = state.graphQuery.trim().toLowerCase();
    const matches = new Set();
    if (query) {
      const includeDescendants = (id) => {
        matches.add(id);
        for (const child of explorer.children.get(id) || []) {
          includeDescendants(child);
        }
      };
      for (const item of explorer.nodes.values()) {
        if (`${item.id} ${item.label} ${item.type}`.toLowerCase().includes(query)) {
          includeDescendants(item.id);
          let current = item.id;
          while (current) {
            matches.add(current);
            current = explorer.parents.get(current);
          }
        }
      }
    }
    const visible = [];
    const edges = [];
    if (state.graphView === "leaves") {
      for (const item of explorer.nodes.values()) {
        if (explorer.children.get(item.id).length) continue;
        if (query && !matches.has(item.id)) continue;
        visible.push({ ...item, displayDepth: 0 });
      }
      return { visible, edges, expanded, query };
    }
    const visit = (id, depth) => {
      if (query && !matches.has(id)) return;
      const item = explorer.nodes.get(id);
      if (!item) return;
      visible.push({ ...item, displayDepth: depth });
      if (!query && !expanded.has(id)) return;
      for (const child of explorer.children.get(id)) {
        if (!query || matches.has(child)) {
          edges.push({ source: id, target: child });
          visit(child, depth + 1);
        }
      }
    };
    explorer.roots.forEach((id) => visit(id, 0));
    return { visible, edges, expanded, query };
  }

  function layoutGraphItems(items) {
    const positions = new Map();
    if (state.graphView === "leaves") {
      const columnCount = Math.min(3, Math.max(1, items.length));
      const rowCount = Math.ceil(items.length / columnCount);
      items.forEach((item, index) => positions.set(item.id, {
        x: 20 + ((index % columnCount) * 218),
        y: 20 + (Math.floor(index / columnCount) * 78),
      }));
      return {
        positions,
        width: Math.max(430, 30 + (columnCount * 218)),
        height: Math.max(330, 30 + (rowCount * 78)),
      };
    }
    const columns = new Map();
    for (const item of items) {
      if (!columns.has(item.displayDepth)) columns.set(item.displayDepth, []);
      columns.get(item.displayDepth).push(item);
    }
    const maxDepth = Math.max(...columns.keys());
    const maxRows = Math.max(...[...columns.values()].map((column) => column.length));
    const width = Math.max(430, 35 + ((maxDepth + 1) * 218));
    const height = Math.max(330, 35 + (maxRows * 78));
    for (const [depth, column] of columns.entries()) {
      const offset = Math.max(20, (height - (column.length * 78)) / 2);
      column.forEach((item, index) => positions.set(item.id, {
        x: 20 + (depth * 218),
        y: offset + (index * 78),
      }));
    }
    return { positions, width, height };
  }

  function operatorExpandedModules(graph) {
    const key = graphExpansionKey(graph);
    if (!Object.hasOwn(state.graphExpandedNodes, key)) {
      state.graphExpandedNodes[key] = ["__root__"];
      persistState();
    }
    return new Set(state.graphExpandedNodes[key]);
  }

  function operatorSearchText(item) {
    return `${item.id} ${item.label} ${item.type} ${item.target || ""} ${item.module_path || ""}`.toLowerCase();
  }

  function buildOperatorExplorer(graph) {
    const rawNodes = new Map(graph.nodes.map((item) => [item.id, item]));
    const modules = new Map((graph.modules || []).map((item) => [item.id, item]));
    if (!modules.has("__root__")) {
      modules.set("__root__", {
        id: "__root__",
        path: "",
        label: graph.summary.model_class,
        type: graph.summary.model_class,
        parameters: graph.summary.parameters,
        trainable_parameters: graph.summary.parameters,
        operator_count: graph.summary.operators,
      });
    }
    const incoming = new Map(graph.nodes.map((item) => [item.id, []]));
    for (const edge of graph.edges) {
      if (incoming.has(edge.target) && rawNodes.has(edge.source)) {
        incoming.get(edge.target).push(edge.source);
      }
    }
    const rawDepths = new Map();
    for (const item of graph.nodes) {
      const parentDepths = incoming.get(item.id).map((parent) => rawDepths.get(parent) || 0);
      rawDepths.set(item.id, parentDepths.length ? Math.max(...parentDepths) + 1 : 0);
    }

    const expanded = operatorExpandedModules(graph);
    const representatives = new Map();
    const visibleMap = new Map();
    const representativeFor = (item) => {
      if (!expanded.has("__root__")) return "module:__root__";
      for (const moduleId of item.module_stack || []) {
        if (modules.has(moduleId) && !expanded.has(moduleId)) {
          return `module:${moduleId}`;
        }
      }
      return `operator:${item.id}`;
    };

    for (const item of graph.nodes) {
      const representativeId = representativeFor(item);
      representatives.set(item.id, representativeId);
      if (!visibleMap.has(representativeId)) {
        if (representativeId.startsWith("module:")) {
          const moduleId = representativeId.slice("module:".length);
          const module = modules.get(moduleId);
          visibleMap.set(representativeId, {
            ...module,
            id: representativeId,
            module_id: moduleId,
            module_path: module.path,
            kind: "module",
            op: "module",
            target: module.path || graph.summary.model_class,
            tensors: [],
            rawIds: [],
            displayDepth: Number.POSITIVE_INFINITY,
          });
        } else {
          visibleMap.set(representativeId, {
            ...item,
            id: representativeId,
            raw_id: item.id,
            kind: "operator",
            rawIds: [],
            displayDepth: rawDepths.get(item.id) || 0,
          });
        }
      }
      visibleMap.get(representativeId).rawIds.push(item.id);
    }

    for (const visibleItem of visibleMap.values()) {
      if (visibleItem.kind !== "module") continue;
      const operationDepths = visibleItem.rawIds
        .map((id) => rawNodes.get(id))
        .filter((item) => item && item.op !== "placeholder")
        .map((item) => rawDepths.get(item.id) || 0);
      const allDepths = visibleItem.rawIds.map((id) => rawDepths.get(id) || 0);
      visibleItem.displayDepth = Math.min(...(operationDepths.length ? operationDepths : allDepths));
    }

    const edgeMap = new Map();
    for (const edge of graph.edges) {
      const source = representatives.get(edge.source);
      const target = representatives.get(edge.target);
      if (!source || !target || source === target) continue;
      const key = `${source}\u0000${target}`;
      if (!edgeMap.has(key)) edgeMap.set(key, { source, target, count: 0 });
      edgeMap.get(key).count += 1;
    }

    const query = state.graphQuery.trim().toLowerCase();
    let matchCount = 0;
    for (const visibleItem of visibleMap.values()) {
      const contributorText = visibleItem.rawIds
        .map((id) => operatorSearchText(rawNodes.get(id)))
        .join(" ");
      visibleItem.queryMatch = !query
        || operatorSearchText(visibleItem).includes(query)
        || contributorText.includes(query);
      if (visibleItem.queryMatch) matchCount += 1;
    }
    return {
      visible: [...visibleMap.values()],
      edges: [...edgeMap.values()],
      expanded,
      modules,
      rawNodes,
      query,
      matchCount,
    };
  }

  function layoutOperatorGraph(items) {
    const positions = new Map();
    const columns = new Map();
    for (const item of items) {
      if (!columns.has(item.displayDepth)) columns.set(item.displayDepth, []);
      columns.get(item.displayDepth).push(item);
    }
    const depths = [...columns.keys()];
    const maxDepth = depths.length ? Math.max(...depths) : 0;
    const maxRows = Math.max(1, ...[...columns.values()].map((column) => column.length));
    const width = Math.max(460, 58 + ((maxDepth + 1) * 228));
    const height = Math.max(360, 72 + (maxRows * 88));
    for (const [depth, column] of columns.entries()) {
      const offset = Math.max(55, (height - (column.length * 88)) / 2);
      column.forEach((item, index) => positions.set(item.id, {
        x: 36 + (depth * 228),
        y: offset + (index * 88),
      }));
    }
    return { positions, width, height };
  }

  function operatorModuleFrames(explorer, positions) {
    const frames = [];
    for (const moduleId of explorer.expanded) {
      if (moduleId === "__root__") continue;
      const module = explorer.modules.get(moduleId);
      if (!module) continue;
      const descendants = explorer.visible.filter((item) => item.rawIds.some((rawId) =>
        explorer.rawNodes.get(rawId)?.module_stack?.includes(moduleId)));
      const points = descendants.map((item) => positions.get(item.id)).filter(Boolean);
      if (!points.length) continue;
      const minX = Math.min(...points.map((point) => point.x));
      const minY = Math.min(...points.map((point) => point.y));
      const maxX = Math.max(...points.map((point) => point.x + 198));
      const maxY = Math.max(...points.map((point) => point.y + 64));
      frames.push({
        module,
        x: minX - 12,
        y: minY - 30,
        width: (maxX - minX) + 24,
        height: (maxY - minY) + 42,
      });
    }
    return frames.sort((left, right) =>
      left.module.path.split(".").length - right.module.path.split(".").length);
  }

  function beginOperatorNodeDrag(event, item, position, zoom, group) {
    if (
      event.button !== 0
      || event.target.closest?.(".graph-expansion-control")
    ) return;
    event.stopPropagation();
    const host = el("model-graph-canvas");
    host.setPointerCapture(event.pointerId);
    group.classList.add("dragging");
    graphPointerDrag = {
      kind: "node",
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startX: position.x,
      startY: position.y,
      currentX: position.x,
      currentY: position.y,
      itemId: item.id,
      zoom,
      group,
      moved: false,
    };
  }

  function beginGraphPan(event) {
    if (
      event.button !== 0
      || event.target.closest?.(".graph-node, .operator-module-frame-header")
    ) return;
    const host = el("model-graph-canvas");
    host.setPointerCapture(event.pointerId);
    host.classList.add("is-panning");
    graphPointerDrag = {
      kind: "pan",
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startScrollLeft: host.scrollLeft,
      startScrollTop: host.scrollTop,
      moved: false,
    };
  }

  function updateGraphPointerDrag(event) {
    if (!graphPointerDrag || graphPointerDrag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - graphPointerDrag.startClientX;
    const deltaY = event.clientY - graphPointerDrag.startClientY;
    graphPointerDrag.moved ||= Math.abs(deltaX) + Math.abs(deltaY) > 3;
    const host = el("model-graph-canvas");
    if (graphPointerDrag.kind === "pan") {
      host.scrollLeft = graphPointerDrag.startScrollLeft - deltaX;
      host.scrollTop = graphPointerDrag.startScrollTop - deltaY;
    } else {
      graphPointerDrag.currentX = Math.max(
        8,
        graphPointerDrag.startX + (deltaX / graphPointerDrag.zoom),
      );
      graphPointerDrag.currentY = Math.max(
        34,
        graphPointerDrag.startY + (deltaY / graphPointerDrag.zoom),
      );
      graphPointerDrag.group.setAttribute(
        "transform",
        `translate(${graphPointerDrag.currentX} ${graphPointerDrag.currentY})`,
      );
    }
    event.preventDefault();
  }

  function finishGraphPointerDrag(event) {
    if (!graphPointerDrag || graphPointerDrag.pointerId !== event.pointerId) return;
    const drag = graphPointerDrag;
    graphPointerDrag = null;
    const host = el("model-graph-canvas");
    host.classList.remove("is-panning");
    drag.group?.classList.remove("dragging");
    if (host.hasPointerCapture(event.pointerId)) {
      host.releasePointerCapture(event.pointerId);
    }
    if (drag.kind === "node" && drag.moved) {
      const graph = state.modelGraphData;
      const key = graphPositionKey(graph);
      state.graphNodePositions[key] = {
        ...(state.graphNodePositions[key] || {}),
        [drag.itemId]: { x: drag.currentX, y: drag.currentY },
      };
      persistState();
      renderModelGraphCanvas();
    }
  }

  function renderOperatorGraphCanvas(graph, host, previousScroll) {
    const explorer = buildOperatorExplorer(graph);
    if (!explorer.visible.length || (explorer.query && !explorer.matchCount)) {
      host.append(node("div", "graph-empty", explorer.query
        ? `No modules or operators match “${state.graphQuery}”.`
        : "The exported program contains no operators."));
      return;
    }
    const { positions, width, height } = applyOperatorPositionOverrides(
      graph,
      explorer.visible,
      layoutOperatorGraph(explorer.visible),
    );
    const zoom = currentGraphZoom(graph);
    const svg = graphSvgElement("svg", {
      width: Math.ceil(width * zoom),
      height: Math.ceil(height * zoom),
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Hierarchical computational graph with ${explorer.visible.length} visible nodes`,
    });
    const markerId = `operator-arrow-${state.graphRequestId}`;
    const definitions = graphSvgElement("defs");
    const marker = graphSvgElement("marker", {
      id: markerId,
      viewBox: "0 0 8 8",
      refX: "7",
      refY: "4",
      markerWidth: "7",
      markerHeight: "7",
      orient: "auto-start-reverse",
    });
    marker.append(graphSvgElement("path", {
      class: "graph-arrow-head",
      d: "M 0 0 L 8 4 L 0 8 z",
    }));
    definitions.append(marker);
    svg.append(definitions);

    const frameLayer = graphSvgElement("g", { class: "operator-module-frames" });
    for (const frame of operatorModuleFrames(explorer, positions)) {
      const frameGroup = graphSvgElement("g", { class: "operator-module-frame" });
      frameGroup.append(graphSvgElement("rect", {
        x: frame.x,
        y: frame.y,
        width: frame.width,
        height: frame.height,
        class: "operator-module-frame-box",
      }));
      const header = graphSvgElement("g", {
        class: "operator-module-frame-header",
        tabindex: "0",
        role: "button",
        "aria-label": `Collapse ${frame.module.path}`,
      });
      header.append(
        graphSvgElement("rect", {
          x: frame.x + 5,
          y: frame.y + 4,
          width: Math.min(frame.width - 10, 178),
          height: 19,
          class: "operator-module-frame-title-box",
        }),
        graphSvgElement("text", {
          x: frame.x + 12,
          y: frame.y + 18,
          class: "operator-module-frame-title",
        }, `− ${frame.module.path}`),
      );
      const collapse = () => toggleOperatorModule(frame.module.id);
      header.addEventListener("click", collapse);
      header.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          collapse();
        }
      });
      frameGroup.append(header);
      frameLayer.append(frameGroup);
    }
    svg.append(frameLayer);

    const edgeLayer = graphSvgElement("g", { class: "graph-edges" });
    for (const edge of explorer.edges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) continue;
      const forward = source.x < target.x;
      const startX = forward ? source.x + 198 : source.x + 99;
      const startY = forward ? source.y + 32 : source.y + 64;
      const endX = forward ? target.x : target.x + 99;
      const endY = forward ? target.y + 32 : target.y;
      const middle = forward ? (startX + endX) / 2 : Math.max(startY, endY) + 26;
      const path = forward
        ? `M ${startX} ${startY} C ${middle} ${startY}, ${middle} ${endY}, ${endX} ${endY}`
        : `M ${startX} ${startY} C ${startX} ${middle}, ${endX} ${middle}, ${endX} ${endY}`;
      edgeLayer.append(graphSvgElement("path", {
        class: "graph-edge operator-edge",
        d: path,
        "marker-end": `url(#${markerId})`,
      }));
    }
    svg.append(edgeLayer);

    const nodeLayer = graphSvgElement("g", { class: "graph-nodes" });
    const modelTotal = Math.max(Number(graph.summary.parameters) || 0, 1);
    for (const item of explorer.visible) {
      const position = positions.get(item.id);
      const parameterShare = Math.min(1, (Number(item.parameters) || 0) / modelTotal);
      const parameterColor = graphParameterColor(parameterShare);
      const searchClass = explorer.query
        ? (item.queryMatch ? " search-match" : " search-muted")
        : "";
      const group = graphSvgElement("g", {
        class: `graph-node operator-node${item.kind === "module" ? " module-node" : ""}${parameterColor.dense ? " dense" : ""}${searchClass}`,
        transform: `translate(${position.x} ${position.y})`,
        tabindex: "0",
        role: "button",
      });
      group.addEventListener("pointerdown", (event) => {
        beginOperatorNodeDrag(event, item, position, zoom, group);
      });
      const label = item.label.length > 28 ? `${item.label.slice(0, 26)}…` : item.label;
      const origin = item.kind === "module" ? item.type : (item.module_path || item.type);
      const detail = item.kind === "module"
        ? `${formatNumber(item.operator_count)} ops · ${formatParameterShare(parameterShare)} params`
        : (item.parameters
          ? `${formatParameterShare(parameterShare)} · ${formatNumber(item.parameters)} params`
          : (item.tensors?.[0]?.shape ? `[${item.tensors[0].shape.join(", ")}]` : item.op));
      const labelX = item.kind === "module" ? 29 : 10;
      group.append(
        graphSvgElement("rect", {
          width: 198,
          height: 64,
          class: "graph-node-card operator-node-card",
          fill: parameterColor.color,
        }),
        graphSvgElement("text", { x: labelX, y: 19, class: "graph-node-label" }, label),
        graphSvgElement("text", { x: 10, y: 38, class: "graph-node-type" }, origin.length > 30 ? `${origin.slice(0, 28)}…` : origin),
        graphSvgElement("text", { x: 10, y: 54, class: "graph-node-share" }, detail),
      );
      if (item.kind === "module") {
        const expansion = graphSvgElement("g", { class: "graph-expansion-control" });
        expansion.append(
          graphSvgElement("rect", {
            x: 8,
            y: 8,
            width: 14,
            height: 14,
            class: "graph-node-expand",
          }),
          graphSvgElement("text", {
            x: 12,
            y: 19,
            class: "graph-node-expand-symbol",
          }, "+"),
        );
        expansion.addEventListener("click", (event) => {
          event.stopPropagation();
          toggleOperatorModule(item.module_id);
        });
        group.append(expansion);
      }
      const select = () => {
        svg.querySelectorAll(".graph-node.selected").forEach(
          (candidate) => candidate.classList.remove("selected"),
        );
        group.classList.add("selected");
        renderModelNodeDetails({ ...item, parameterShare });
      };
      group.addEventListener("click", select);
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
      nodeLayer.append(group);
    }
    svg.append(nodeLayer);
    host.append(svg);
    host.scrollLeft = previousScroll.left;
    host.scrollTop = previousScroll.top;
    syncGraphZoomControls();
  }

  function renderModelGraphCanvas() {
    const graph = state.modelGraphData;
    const host = el("model-graph-canvas");
    const previousScroll = { left: host.scrollLeft, top: host.scrollTop };
    host.replaceChildren();
    if (!graph?.nodes.length) {
      host.append(node("div", "graph-empty", "The checkpoint contains no graphable modules."));
      return;
    }
    if (graph.graph_mode === "operators") {
      renderOperatorGraphCanvas(graph, host, previousScroll);
      return;
    }
    const explorer = buildGraphExplorerModel(graph);
    const { visible, edges, expanded, query } = visibleGraphItems(graph, explorer);
    if (!visible.length) {
      host.append(node("div", "graph-empty", query
        ? `No model blocks match “${state.graphQuery}”.`
        : "No leaf modules are available."));
      return;
    }
    const { positions, width, height } = layoutGraphItems(visible);
    const svg = graphSvgElement("svg", {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
    });
    const edgeLayer = graphSvgElement("g", { class: "graph-edges" });
    for (const edge of edges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) continue;
      const startX = source.x + 190;
      const startY = source.y + 29;
      const endX = target.x;
      const endY = target.y + 29;
      const middle = (startX + endX) / 2;
      edgeLayer.append(graphSvgElement("path", {
        class: "graph-edge",
        d: `M ${startX} ${startY} C ${middle} ${startY}, ${middle} ${endY}, ${endX} ${endY}`,
      }));
    }
    svg.append(edgeLayer);
    const nodeLayer = graphSvgElement("g", { class: "graph-nodes" });
    for (const item of visible) {
      const position = positions.get(item.id);
      const parameterColor = graphParameterColor(item.parameterShare);
      const hasChildren = explorer.children.get(item.id).length > 0;
      const isExpanded = expanded.has(item.id) || Boolean(query);
      const group = graphSvgElement("g", {
        class: `graph-node${parameterColor.dense ? " dense" : ""}`,
        transform: `translate(${position.x} ${position.y})`,
        tabindex: "0",
        role: "button",
      });
      const labelX = hasChildren && state.graphView === "hierarchy" ? 29 : 10;
      group.append(
        graphSvgElement("rect", {
          width: 190,
          height: 58,
          class: "graph-node-card",
          fill: parameterColor.color,
        }),
        graphSvgElement("text", {
          x: labelX,
          y: 19,
          class: "graph-node-label",
        }, item.label.length > 24 ? `${item.label.slice(0, 22)}…` : item.label),
        graphSvgElement("text", {
          x: 10,
          y: 36,
          class: "graph-node-type",
        }, item.type),
        graphSvgElement("text", {
          x: 10,
          y: 50,
          class: "graph-node-share",
        }, `${formatParameterShare(item.parameterShare)} · ${formatNumber(item.blockParameters)} params`),
      );
      if (hasChildren && state.graphView === "hierarchy") {
        const expansion = graphSvgElement("g", {
          class: "graph-expansion-control",
        });
        expansion.append(
          graphSvgElement("rect", {
            x: 8,
            y: 8,
            width: 14,
            height: 14,
            class: "graph-node-expand",
          }),
          graphSvgElement("text", {
            x: 12,
            y: 19,
            class: "graph-node-expand-symbol",
          }, isExpanded ? "−" : "+"),
        );
        expansion.addEventListener("click", (event) => {
          event.stopPropagation();
          toggleGraphBlock(item.id);
        });
        group.append(expansion);
      }
      const select = () => {
        svg.querySelectorAll(".graph-node.selected").forEach(
          (candidate) => candidate.classList.remove("selected"),
        );
        group.classList.add("selected");
        renderModelNodeDetails(item);
      };
      group.addEventListener("click", select);
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") select();
      });
      nodeLayer.append(group);
    }
    svg.append(nodeLayer);
    host.append(svg);
    host.scrollLeft = previousScroll.left;
    host.scrollTop = previousScroll.top;
  }

  function formatParameterShare(share) {
    const percentage = Math.max(0, Number(share) || 0) * 100;
    return `${percentage.toFixed(percentage < 0.1 ? 3 : 1)}%`;
  }

  function toggleGraphBlock(nodeId) {
    const graph = state.modelGraphData;
    if (!graph) return;
    const key = graphExpansionKey(graph);
    const expanded = new Set(state.graphExpandedNodes[key] || []);
    if (expanded.has(nodeId)) expanded.delete(nodeId); else expanded.add(nodeId);
    state.graphExpandedNodes[key] = [...expanded];
    persistState();
    renderModelGraphCanvas();
  }

  function toggleOperatorModule(moduleId) {
    const graph = state.modelGraphData;
    if (!graph || graph.graph_mode !== "operators") return;
    const key = graphExpansionKey(graph);
    const expanded = operatorExpandedModules(graph);
    if (expanded.has(moduleId)) {
      const modulePath = (graph.modules || []).find((item) => item.id === moduleId)?.path || "";
      for (const candidate of [...expanded]) {
        const candidatePath = (graph.modules || []).find((item) => item.id === candidate)?.path || "";
        if (candidate === moduleId || !modulePath || candidatePath.startsWith(`${modulePath}.`)) {
          expanded.delete(candidate);
        }
      }
    } else {
      expanded.add(moduleId);
    }
    state.graphExpandedNodes[key] = [...expanded];
    persistState();
    renderModelGraphCanvas();
  }

  function setAllGraphBlocks(expand) {
    const graph = state.modelGraphData;
    if (!graph) return;
    const key = graphExpansionKey(graph);
    if (graph.graph_mode === "operators") {
      state.graphExpandedNodes[key] = expand
        ? (graph.modules || []).map((item) => item.id)
        : [];
    } else {
      const explorer = buildGraphExplorerModel(graph);
      state.graphExpandedNodes[key] = expand
        ? [...explorer.nodes.keys()].filter(
          (id) => explorer.children.get(id).length,
        )
        : [];
    }
    persistState();
    renderModelGraphCanvas();
  }

  function renderGraphModeSelect(info) {
    const select = el("model-graph-mode-select");
    const operatorOption = [...select.options].find((option) => option.value === "operators");
    const operatorMode = info.modes?.operators || { available: false, reason: "Operators view is unavailable for this run." };
    operatorOption.disabled = !operatorMode.available;
    operatorOption.textContent = operatorMode.available ? "Operators" : "Operators · unavailable";
    select.title = operatorMode.available ? "" : operatorMode.reason;
    if (state.graphMode === "operators" && !operatorMode.available) {
      state.graphMode = "architecture";
      persistState();
    }
    select.value = state.graphMode;
    select.disabled = false;
  }

  function renderGraphCheckpointSelect(graph) {
    const select = el("model-graph-checkpoint-select");
    select.replaceChildren();
    const auto = node("option", "", `Auto (${graph.checkpoint.kind || "unavailable"})`);
    auto.value = "auto";
    select.append(auto);
    for (const kind of graph.checkpoint.available) {
      const size = graph.checkpoint.sizes_mb?.[kind];
      const tooLarge = !graph.checkpoint.loadable.includes(kind);
      const suffix = size === undefined ? "" : ` · ${formatNumber(size)} MB${tooLarge ? " · too large" : ""}`;
      const option = node("option", "", `${kind[0].toUpperCase()}${kind.slice(1)}${suffix}`);
      option.value = kind;
      option.disabled = tooLarge;
      select.append(option);
    }
    select.value = state.graphCheckpointChoices[graph.run] || "auto";
    select.disabled = false;
  }

  function renderModelNodeDetails(item) {
    const panel = el("model-node-details");
    panel.replaceChildren();
    panel.append(node("h4", "", item.label), node("code", "", item.id));
    const values = node("dl", "");
    const operator = state.modelGraphData?.graph_mode === "operators";
    const fields = operator && item.kind === "module" ? [
      ["Module type", item.type],
      ["Module path", item.module_path || "Model root"],
      ["Contained operators", formatNumber(item.operator_count)],
      ["Block parameters", formatNumber(item.parameters)],
      ["Share of model", formatParameterShare(item.parameterShare)],
      ["Trainable", formatNumber(item.trainable_parameters)],
    ] : operator ? [
      ["Node type", item.type],
      ["Operation", item.op],
      ["Target", item.target],
      ["Origin module", item.module_path || "—"],
      ["Parameters represented", formatNumber(item.parameters)],
      ["Share of model", formatParameterShare(item.parameterShare)],
    ] : [
      ["Module type", item.type],
      ["Block parameters", formatNumber(item.blockParameters ?? item.parameters)],
      ["Share of model", formatParameterShare(item.parameterShare)],
      ["Direct parameters", formatNumber(item.parameters)],
      ["Trainable", formatNumber(item.trainable_parameters)],
    ];
    for (const [label, value] of fields) {
      values.append(node("dt", "", label), node("dd", "", value));
    }
    panel.append(values);
    if (item.tensors?.length) {
      panel.append(node("strong", "", operator ? "Exported tensors" : "Checkpoint tensors"));
      const list = node("ul", "");
      for (const tensor of item.tensors) {
        const dtype = tensor.dtype ? ` · ${tensor.dtype}` : "";
        list.append(node("li", "", `${tensor.name} · [${tensor.shape.join(", ")}]${dtype}`));
      }
      panel.append(list);
    }
  }

  function updateChartHover(chart, event) {
    const rect = chart.canvas.getBoundingClientRect();
    const margin = { right: 12, left: 48 };
    const plotWidth = rect.width - margin.left - margin.right;
    const pointerX = event.clientX - rect.left;
    if (pointerX < margin.left || pointerX > rect.width - margin.right) {
      if (chart.hoverIndex !== null) {
        chart.hoverIndex = null;
        updateChartReadout(chart);
        drawChart(chart);
      }
      return;
    }
    const epochs = Math.max(...chart.group.lines.map((line) => line.values.length), 1);
    const index = epochs <= 1
      ? 0
      : Math.round(((pointerX - margin.left) / plotWidth) * (epochs - 1));
    if (index !== chart.hoverIndex) {
      chart.hoverIndex = Math.max(0, Math.min(epochs - 1, index));
      updateChartReadout(chart);
      drawChart(chart);
    }
  }

  function updateChartReadout(chart) {
    chart.epochLabel.textContent = chart.hoverIndex === null
      ? "Latest"
      : `Epoch ${chart.hoverIndex + 1}`;
    for (const line of chart.group.lines) {
      chart.legendValues.get(line.id).textContent = formatLineReadout(
        line,
        chart.hoverIndex,
      );
    }
  }

  function formatLineReadout(line, requestedIndex = null) {
    let index = requestedIndex;
    if (index === null) {
      index = line.values.length - 1;
      while (index >= 0 && (line.values[index] === null || !Number.isFinite(line.values[index]))) index -= 1;
    }
    const value = index >= 0 ? line.values[index] : undefined;
    if (!line.band || !Number.isFinite(value)) return formatNumber(value);
    const lower = line.band.lower[index];
    const upper = line.band.upper[index];
    const std = Number.isFinite(lower) && Number.isFinite(upper)
      ? (upper - lower) / 2
      : undefined;
    return `${formatNumber(value)} ± ${formatNumber(std)}`;
  }

  function createValueScale(values) {
    if (state.scale !== "symlog") {
      return { transform: (value) => value, invert: (value) => value };
    }
    let maxMagnitude = 1;
    let minMagnitude = Infinity;
    for (const value of values) {
      const magnitude = Math.abs(value);
      if (magnitude > 0) {
        maxMagnitude = Math.max(maxMagnitude, magnitude);
        minMagnitude = Math.min(minMagnitude, magnitude);
      }
    }
    if (!Number.isFinite(minMagnitude)) minMagnitude = maxMagnitude;
    const linearThreshold = Math.max(minMagnitude, maxMagnitude * 1e-6, 1e-12);
    return {
      transform: (value) => Math.sign(value) * Math.log10(1 + Math.abs(value) / linearThreshold),
      invert: (value) => Math.sign(value) * linearThreshold * (10 ** Math.abs(value) - 1),
    };
  }

  function drawMetricBand(ctx, line, x, y) {
    if (!line.band) return;
    const { lower, upper } = line.band;
    let start = null;
    const drawSegment = (from, to) => {
      ctx.beginPath();
      ctx.moveTo(x(from), y(lower[from]));
      for (let index = from + 1; index <= to; index += 1) ctx.lineTo(x(index), y(lower[index]));
      for (let index = to; index >= from; index -= 1) ctx.lineTo(x(index), y(upper[index]));
      ctx.closePath();
      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = colors[line.split] || colors.other;
      ctx.fill();
      ctx.restore();
    };
    for (let index = 0; index <= lower.length; index += 1) {
      const finite = index < lower.length
        && Number.isFinite(lower[index])
        && Number.isFinite(upper[index]);
      if (finite && start === null) start = index;
      if (!finite && start !== null) {
        drawSegment(start, index - 1);
        start = null;
      }
    }
  }

  function drawChart(chart) {
    const { canvas, group } = chart;
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
    const themeStyles = getComputedStyle(document.documentElement);
    const gridColor = themeStyles.getPropertyValue("--chart-grid").trim() || "#edf0ed";
    const labelColor = themeStyles.getPropertyValue("--chart-label").trim() || "#8a94a1";
    const guideColor = themeStyles.getPropertyValue("--chart-guide").trim() || "#b5bec8";
    const dotCenter = themeStyles.getPropertyValue("--panel").trim() || "#ffffff";
    const values = group.lines.flatMap((line) => [
      ...line.values,
      ...(line.band?.lower || []),
      ...(line.band?.upper || []),
    ]).filter((value) => value !== null && Number.isFinite(value));
    if (!values.length) return;
    const valueScale = createValueScale(values);
    let min = Infinity;
    let max = -Infinity;
    for (const value of values) {
      const transformed = valueScale.transform(value);
      min = Math.min(min, transformed);
      max = Math.max(max, transformed);
    }
    if (min === max) { min -= Math.abs(min || 1) * 0.05; max += Math.abs(max || 1) * 0.05; }
    const padding = (max - min) * 0.08;
    min -= padding;
    max += padding;
    const epochs = Math.max(...group.lines.map((line) => line.values.length), 1);
    const x = (index) => margin.left + (epochs <= 1 ? plotWidth / 2 : (index / (epochs - 1)) * plotWidth);
    const y = (value) => margin.top + ((max - valueScale.transform(value)) / (max - min)) * plotHeight;

    ctx.font = '9px Inter, -apple-system, sans-serif';
    ctx.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const transformedValue = min + ((max - min) * tick) / 4;
      const value = valueScale.invert(transformedValue);
      const tickY = margin.top + ((max - transformedValue) / (max - min)) * plotHeight;
      ctx.beginPath(); ctx.moveTo(margin.left, tickY); ctx.lineTo(width - margin.right, tickY);
      ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = labelColor; ctx.textAlign = "right";
      ctx.fillText(formatNumber(value), margin.left - 7, tickY);
    }
    ctx.fillStyle = labelColor; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("1", x(0), height - margin.bottom + 8);
    if (epochs > 1) ctx.fillText(String(epochs), x(epochs - 1), height - margin.bottom + 8);
    ctx.fillStyle = labelColor; ctx.fillText("epoch", margin.left + plotWidth / 2, height - 10);

    for (const line of group.lines) drawMetricBand(ctx, line, x, y);

    for (const line of group.lines) {
      ctx.beginPath();
      let drawing = false;
      line.values.forEach((value, index) => {
        if (value === null || !Number.isFinite(value)) { drawing = false; return; }
        if (!drawing) { ctx.moveTo(x(index), y(value)); drawing = true; } else { ctx.lineTo(x(index), y(value)); }
      });
      ctx.strokeStyle = colors[line.split] || colors.other;
      ctx.lineWidth = 2;
      ctx.setLineDash(line.dash || []);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    }
    ctx.setLineDash([]);

    if (chart.hoverIndex !== null) {
      const hoverX = x(chart.hoverIndex);
      ctx.save();
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(hoverX, margin.top);
      ctx.lineTo(hoverX, margin.top + plotHeight);
      ctx.strokeStyle = guideColor;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
      for (const line of group.lines) {
        const value = line.values[chart.hoverIndex];
        if (value === null || !Number.isFinite(value)) continue;
        ctx.beginPath();
        ctx.arc(hoverX, y(value), 4.2, 0, Math.PI * 2);
        ctx.fillStyle = dotCenter;
        ctx.fill();
        ctx.lineWidth = 2.4;
        ctx.strokeStyle = colors[line.split] || colors.other;
        ctx.stroke();
      }
    }
  }

  function jsonValueKind(value) {
    if (value === null) return "null";
    if (Array.isArray(value)) return "array";
    return typeof value;
  }

  function jsonCollectionLabel(value) {
    if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? "" : "s"}`;
    const count = Object.keys(value).length;
    return `${count} field${count === 1 ? "" : "s"}`;
  }

  function isNestedBelowConfig(jsonPath) {
    const configIndex = jsonPath.indexOf("config");
    return configIndex !== -1 && configIndex < jsonPath.length - 1;
  }

  function expandJsonDescendants(collection) {
    collection.querySelectorAll("details.json-collection").forEach((descendant) => {
      descendant.open = true;
    });
  }

  function renderJsonValue(value, key, disclosurePrefix, jsonPath = []) {
    const kind = jsonValueKind(value);
    const row = node("div", "json-row");

    if (kind !== "object" && kind !== "array") {
      if (key !== null) row.append(node("span", "json-key", String(key)));
      let display = String(value);
      if (kind === "string") display = value;
      row.append(node("span", `json-value json-${kind}`, display));
      return row;
    }

    const entries = Array.isArray(value) ? value.entries() : Object.entries(value);
    const collection = node("details", `json-collection json-${kind}`);
    bindDisclosure(
      collection,
      `${disclosurePrefix}:${JSON.stringify(jsonPath)}`,
      key === null,
    );
    const summary = node("summary", "json-collection-summary");
    if (key !== null) summary.append(node("span", "json-key", String(key)));
    summary.append(node("span", "json-count", jsonCollectionLabel(value)));
    collection.append(summary);

    const children = node("div", "json-children");
    for (const [childKey, childValue] of entries) {
      children.append(renderJsonValue(
        childValue,
        childKey,
        disclosurePrefix,
        [...jsonPath, childKey],
      ));
    }
    if (!children.childElementCount) children.append(node("span", "json-empty", "Empty"));
    collection.append(children);
    if (isNestedBelowConfig(jsonPath)) {
      collection.addEventListener("toggle", () => {
        if (collection.open) expandJsonDescendants(collection);
      });
    }
    row.append(collection);
    return row;
  }

  function metadataViewer(item) {
    const body = node("div", "metadata-body");
    const toolbar = node("div", "metadata-toolbar");
    const structuredButton = node("button", "active", "Inspector");
    const rawButton = node("button", "", "Raw JSON");
    structuredButton.type = rawButton.type = "button";
    structuredButton.setAttribute("aria-pressed", "true");
    rawButton.setAttribute("aria-pressed", "false");
    toolbar.append(structuredButton, rawButton);

    const structured = node("div", "json-inspector");
    structured.append(renderJsonValue(item.data, null, `metadata-json:${item.path}`));
    const raw = node("pre", "metadata-raw", JSON.stringify(item.data, null, 2));
    raw.hidden = true;

    const modeKey = item.path;
    const inspectorScrollKey = `${item.path}:inspector`;
    const rawScrollKey = `${item.path}:raw`;
    const trackScroll = (element, key) => {
      element.addEventListener("scroll", () => {
        state.metadataScrolls[key] = element.scrollTop;
        clearTimeout(metadataScrollTimer);
        metadataScrollTimer = setTimeout(persistState, 100);
      });
    };
    const restoreScroll = (element, key) => {
      requestAnimationFrame(() => {
        element.scrollTop = Number(state.metadataScrolls[key]) || 0;
      });
    };
    trackScroll(structured, inspectorScrollKey);
    trackScroll(raw, rawScrollKey);

    const setMode = (showRaw, save = true) => {
      structured.hidden = showRaw;
      raw.hidden = !showRaw;
      structuredButton.classList.toggle("active", !showRaw);
      rawButton.classList.toggle("active", showRaw);
      structuredButton.setAttribute("aria-pressed", String(!showRaw));
      rawButton.setAttribute("aria-pressed", String(showRaw));
      restoreScroll(
        showRaw ? raw : structured,
        showRaw ? rawScrollKey : inspectorScrollKey,
      );
      if (save) {
        state.metadataModes[modeKey] = showRaw ? "raw" : "inspector";
        persistState();
      }
    };
    structuredButton.addEventListener("click", () => setMode(false));
    rawButton.addEventListener("click", () => setMode(true));
    setMode(state.metadataModes[modeKey] === "raw", false);
    body.append(toolbar, structured, raw);
    return body;
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
      let viewer = null;
      const ensureViewer = () => {
        if (!viewer) {
          viewer = metadataViewer(item);
          details.append(viewer);
        }
      };
      bindDisclosure(details, `metadata:${item.path}`, false);
      details.addEventListener("toggle", () => {
        if (details.open) ensureViewer();
      });
      details.append(summary);
      if (details.open) ensureViewer();
      list.append(details);
    }
  }

  el("refresh-button").addEventListener("click", () => loadTree());
  el("export-button").addEventListener("click", exportView);
  el("cache-apply").addEventListener("click", applyCacheLimit);
  el("cache-reset").addEventListener("click", resetCache);
  el("cache-limit").addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyCacheLimit();
  });
  el("refresh-apply").addEventListener("click", applyRefreshInterval);
  el("refresh-interval").addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyRefreshInterval();
  });
  el("theme-toggle").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "day" : "dark";
    persistState();
    applyTheme();
    state.charts.forEach((chart) => drawChart(chart));
    if (state.modelGraphData) renderModelGraphCanvas();
  });
  el("metric-search").addEventListener("input", (event) => {
    state.query = event.target.value.trim().toLowerCase();
    persistState();
    renderCharts();
  });
  el("source-select").addEventListener("change", (event) => {
    state.source = event.target.value;
    persistState();
    renderCharts();
  });
  el("plot-mode-select").addEventListener("change", async (event) => {
    state.plotMode = event.target.value;
    state.source = "all";
    persistState();
    if (state.details?.selection.plot_scope === "final_runs") {
      await loadDetails(state.selectedPath, { preserveScroll: true, quiet: true });
      return;
    }
    renderInnerFoldAggregation();
    renderSourceFilter([...new Set(state.details.series.map((series) => series.source))]);
    renderPlotNavigator();
    renderCharts();
    prepareModelGraph();
  });
  el("inner-fold-aggregate").addEventListener("change", (event) => {
    state.innerFoldAggregate = event.target.checked;
    persistState();
    renderCharts();
  });
  el("fold-select").addEventListener("change", (event) => {
    state.focusedInnerFold = event.target.value;
    state.focusedRun = null;
    persistState();
    renderPlotNavigator();
    renderChartsPreservingScroll();
    prepareModelGraph();
  });
  el("run-select").addEventListener("change", (event) => {
    state.focusedRun = event.target.value;
    persistState();
    renderPlotNavigator();
    renderChartsPreservingScroll();
    prepareModelGraph();
  });
  el("fold-previous").addEventListener("click", () => moveNavigatorSelection("fold-select", -1));
  el("fold-next").addEventListener("click", () => moveNavigatorSelection("fold-select", 1));
  el("run-previous").addEventListener("click", () => moveNavigatorSelection("run-select", -1));
  el("run-next").addEventListener("click", () => moveNavigatorSelection("run-select", 1));
  el("show-all-plots").addEventListener("change", (event) => {
    state.showAllPlots = event.target.checked;
    persistState();
    renderPlotNavigator();
    renderChartsPreservingScroll();
  });
  document.querySelectorAll("[data-group]").forEach((button) => {
    button.addEventListener("click", () => {
      state.group = button.dataset.group;
      persistState();
      document.querySelectorAll("[data-group]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      renderCharts();
    });
  });
  el("scale-toggle").addEventListener("click", () => {
    state.scale = state.scale === "symlog" ? "linear" : "symlog";
    persistState();
    syncScaleButton();
    state.charts.forEach((chart) => drawChart(chart));
  });
  let scrollTimer;
  treeElement.addEventListener("scroll", () => {
    state.treeScrollTop = treeElement.scrollTop;
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(persistState, 100);
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.charts.forEach((chart) => drawChart(chart)), 100);
  });

  el("metric-search").value = state.query;
  el("refresh-interval").value = String(state.refreshSeconds);
  updateRefreshStatus();
  applyTheme();
  document.querySelectorAll("[data-group]").forEach((button) => {
    button.classList.toggle("active", button.dataset.group === state.group);
  });
  syncScaleButton();
  observeStickyPlotNavigator();
  bindDisclosure(el("model-graph-section"), "model-graph", false);
  el("model-graph-section").addEventListener("toggle", () => {
    if (el("model-graph-section").open) prepareModelGraph();
  });
  el("model-graph-mode-select").addEventListener("change", (event) => {
    state.graphMode = event.target.value;
    persistState();
    const path = focusedModelGraphPath();
    if (path) loadModelGraph(path);
  });
  el("model-graph-checkpoint-select").addEventListener("change", (event) => {
    const path = focusedModelGraphPath();
    if (!path) return;
    state.graphCheckpointChoices[path] = event.target.value;
    persistState();
    loadModelGraph(path);
  });
  el("model-graph-run-select").addEventListener("change", (event) => {
    const selection = state.details?.selection;
    if (!selection) return;
    state.graphFocusedRuns[graphRunKey(selection)] = event.target.value;
    persistState();
    prepareModelGraph();
  });
  el("graph-expand-all").addEventListener("click", () => setAllGraphBlocks(true));
  el("graph-collapse-all").addEventListener("click", () => setAllGraphBlocks(false));
  el("graph-zoom-out").addEventListener("click", () => {
    setGraphZoom(currentGraphZoom() / 1.2);
  });
  el("graph-zoom-reset").addEventListener("click", () => setGraphZoom(1));
  el("graph-zoom-in").addEventListener("click", () => {
    setGraphZoom(currentGraphZoom() * 1.2);
  });
  el("model-graph-canvas").addEventListener("pointerdown", beginGraphPan);
  el("model-graph-canvas").addEventListener("pointermove", updateGraphPointerDrag);
  el("model-graph-canvas").addEventListener("pointerup", finishGraphPointerDrag);
  el("model-graph-canvas").addEventListener("pointercancel", finishGraphPointerDrag);
  el("model-graph-canvas").addEventListener("wheel", (event) => {
    if (state.modelGraphData?.graph_mode !== "operators") return;
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    setGraphZoom(
      currentGraphZoom() * (event.deltaY < 0 ? 1.1 : (1 / 1.1)),
      { x: event.clientX - bounds.left, y: event.clientY - bounds.top },
    );
  }, { passive: false });
  el("graph-view-toggle").addEventListener("click", () => {
    state.graphView = state.graphView === "hierarchy" ? "leaves" : "hierarchy";
    persistState();
    syncGraphExplorerControls();
    renderModelGraphCanvas();
  });
  el("graph-search").addEventListener("input", (event) => {
    state.graphQuery = event.target.value;
    renderModelGraphCanvas();
  });
  if (state.selectedPath) {
    detailsView.hidden = false;
    welcome.hidden = true;
    el("selection-kind").textContent = "Loading metrics";
    el("selection-name").textContent = state.selectedPath.split("/").pop();
    el("selection-path").textContent = state.selectedPath;
  }

  async function bootstrap() {
    try {
      const imported = await getJson("/api/snapshot-state");
      if (imported && Object.keys(imported).length) {
        Object.assign(state, imported, {
          tree: null,
          details: null,
          charts: [],
          modelGraphData: null,
        });
        el("metric-search").value = state.query || "";
        el("refresh-interval").value = String(state.refreshSeconds || 15);
        applyTheme();
        syncScaleButton();
        persistState();
      }
    } catch (_error) {
      // A live results dashboard has no imported state.
    }
    loadCacheStatus();
    loadTree();
    scheduleRefresh();
  }
  bootstrap();

  function syncScaleButton() {
    const button = el("scale-toggle");
    const active = state.scale === "symlog";
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.textContent = active ? "± Log scale · on" : "± Log scale";
  }
})();
