(() => {
  "use strict";

  // Keep the first occurrence of tied extrema and ignore gaps/non-finite samples.
  function find(values, xValues = null) {
    let min = null;
    let max = null;
    values.forEach((value, index) => {
      const x = xValues ? xValues[index] : index + 1;
      if (!Number.isFinite(value) || !Number.isFinite(x)) return;
      const point = { value, index, x };
      if (!min || value < min.value) min = point;
      if (!max || value > max.value) max = point;
    });
    if (!min) return [];
    if (min.value === max.value) return [{ ...min, kind: "min = max" }];
    return [{ ...min, kind: "min" }, { ...max, kind: "max" }];
  }

  function paired(values, otherValues) {
    return values.map((value, index) => (
      Number.isFinite(value) && Number.isFinite(otherValues[index]) ? value : null
    ));
  }

  function summary(values, format, xValues = null) {
    return find(values, xValues).map((point) => `${point.kind} ${format(point.value)}`).join(" · ");
  }

  function draw(ctx, values, pointAt, { color, opacity = 1, format, xValues = null, prefix = "", bounds }) {
    const annotations = [];
    for (const extremum of find(values, xValues)) {
      const point = pointAt(extremum.index, extremum.value);
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      if (point.x < bounds.left || point.x > bounds.right || point.y < bounds.top || point.y > bounds.bottom) continue;
      const label = `${prefix}${extremum.kind} ${format(extremum.value)}`;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
      const textWidth = ctx.measureText(label).width;
      const labelX = Math.max(bounds.left, Math.min(point.x + 7, bounds.right - textWidth));
      const labelY = Math.max(bounds.top + 8, Math.min(point.y + (extremum.kind === "max" ? 13 : -13), bounds.bottom - 8));
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || "#fff";
      ctx.strokeText(label, labelX, labelY);
      ctx.fillText(label, labelX, labelY);
      ctx.restore();
      annotations.push({ ...extremum, ...point, label });
    }
    return annotations;
  }

  globalThis.MLWizCurveExtrema = { find, paired, summary, draw };
})();
