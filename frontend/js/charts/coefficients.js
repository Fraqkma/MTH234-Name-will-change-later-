export function renderCoefficientChart(target, coefficients) {
  const labels = coefficients.ols.map((_, index) => `w${subscript(index)}`);
  return Plotly.react(target, [
    { x: labels, y: coefficients.ols, type: "bar", name: "OLS", marker: { color: "#287b72" }, hovertemplate: "%{x}: %{y:.3f}<extra>OLS</extra>" },
    { x: labels, y: coefficients.ridge, type: "bar", name: "Ridge", marker: { color: "#b44f32" }, hovertemplate: "%{x}: %{y:.3f}<extra>Ridge</extra>" },
  ], {
    autosize: true,
    height: 256,
    margin: { l: 42, r: 12, t: 8, b: 44 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Inter, system-ui, sans-serif", size: 10, color: "#4f5b56" },
    barmode: "group",
    bargap: 0.24,
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.15, font: { size: 10 } },
    xaxis: { title: { text: "Scaled-basis coefficient", standoff: 4 }, tickfont: { size: 10 }, linecolor: "rgba(31, 38, 35, 0.09)", automargin: true },
    yaxis: { title: { text: "Coefficient", standoff: 6 }, zeroline: true, zerolinecolor: "rgba(31, 38, 35, 0.18)", gridcolor: "rgba(31, 38, 35, 0.09)", automargin: true },
    uirevision: "seeded-coefficients",
  }, { displayModeBar: false, responsive: true, scrollZoom: false });
}

function subscript(value) {
  return String(value).replace(/\d/g, (digit) => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]);
}
