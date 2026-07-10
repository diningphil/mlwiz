(() => {
  "use strict";

  const storageKey = "mlwiz-dashboard-navigation-v1";

  function readStoredState() {
    try {
      return JSON.parse(sessionStorage.getItem(storageKey) || "{}");
    } catch (_error) {
      return {};
    }
  }

  const storedState = readStoredState();
  const storedRefreshSeconds = Number(storedState.refreshSeconds);
  const state = {
    tree: null,
    details: null,
    selectedPath: storedState.selectedPath || null,
    openNodes: storedState.openNodes || {},
    treeScrollTop: Number(storedState.treeScrollTop) || 0,
    group: storedState.group || "all",
    source: storedState.source || "all",
    query: storedState.query || "",
    scale: storedState.scale || "linear",
    theme: ["dark", "day"].includes(storedState.theme) ? storedState.theme : "dark",
    refreshSeconds: Number.isFinite(storedRefreshSeconds)
      && storedRefreshSeconds >= 2
      && storedRefreshSeconds <= 3600
      ? Math.round(storedRefreshSeconds)
      : 15,
    experimentFilters: storedState.experimentFilters || {},
    overviewExpanded: storedState.overviewExpanded !== false,
    filterData: {},
    filterLoading: {},
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
  let refreshTimer = null;

  function persistState() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({
        selectedPath: state.selectedPath,
        openNodes: state.openNodes,
        treeScrollTop: state.treeScrollTop,
        group: state.group,
        source: state.source,
        query: state.query,
        scale: state.scale,
        theme: state.theme,
        refreshSeconds: state.refreshSeconds,
        experimentFilters: state.experimentFilters,
        overviewExpanded: state.overviewExpanded,
      }));
    } catch (_error) {
      // The dashboard remains fully usable when browser storage is disabled.
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
    if (selectionChanged) state.source = "all";
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
      state.details = await getJson(`/api/details?path=${encodeURIComponent(path)}`);
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

  function renderSourceFilter(sources) {
    const field = el("source-field");
    const select = el("source-select");
    select.replaceChildren();
    if (state.source !== "all" && !sources.includes(state.source)) {
      state.source = "all";
      persistState();
    }
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
        swatch.style.background = colors[line.split] || colors.other;
        const value = node("span", "legend-value", formatNumber(lastFiniteValue(line.values)));
        legendValues.set(line.name, value);
        item.append(swatch, document.createTextNode(`${line.split} `), value);
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
      let value;
      if (chart.hoverIndex === null) {
        value = lastFiniteValue(line.values);
      } else {
        value = line.values[chart.hoverIndex];
      }
      chart.legendValues.get(line.name).textContent = formatNumber(value);
    }
  }

  function lastFiniteValue(values) {
    for (let index = values.length - 1; index >= 0; index -= 1) {
      const value = values[index];
      if (value !== null && Number.isFinite(value)) return value;
    }
    return undefined;
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
    const values = group.lines.flatMap((line) => line.values).filter((value) => value !== null && Number.isFinite(value));
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
    }

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
  el("cache-apply").addEventListener("click", applyCacheLimit);
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
  if (state.selectedPath) {
    detailsView.hidden = false;
    welcome.hidden = true;
    el("selection-kind").textContent = "Loading metrics";
    el("selection-name").textContent = state.selectedPath.split("/").pop();
    el("selection-path").textContent = state.selectedPath;
  }

  loadCacheStatus();
  loadTree();
  scheduleRefresh();

  function syncScaleButton() {
    const button = el("scale-toggle");
    const active = state.scale === "symlog";
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.textContent = active ? "± Log scale · on" : "± Log scale";
  }
})();
