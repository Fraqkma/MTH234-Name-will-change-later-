import { fetchDemo } from "./api.js";
import { initBikeExploratoryRanking } from "./bike-rank.js?v=final2";
import { renderCoefficientChart } from "./charts/coefficients.js";
import { renderFitChart } from "./charts/fits.js";
import { getDemoState } from "./state.js";

const degreeInput = document.querySelector("#degree");
const lambdaInput = document.querySelector("#lambda");
const zeroInput = document.querySelector("#no-regularization");
const degreeValue = document.querySelector("#degree-value");
const lambdaValue = document.querySelector("#lambda-value");
const chartState = document.querySelector("#chart-state");
const chartError = document.querySelector("#plot-error");
const connectionState = document.querySelector("#connection-state");
let requestSequence = 0;
let debounce;

function displayLambda(value) {
  if (value === 0) return "0";
  return new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 }).format(value);
}

function updateControlLabels() {
  degreeValue.value = degreeInput.value;
  degreeValue.textContent = degreeInput.value;
  lambdaInput.disabled = zeroInput.checked;
  const lambda = getDemoState().lambda;
  lambdaValue.value = displayLambda(lambda);
  lambdaValue.textContent = displayLambda(lambda);
}

function formatMetric(value) {
  return new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 }).format(value);
}

function setMetric(id, value) {
  document.getElementById(id).textContent = formatMetric(value);
}

function appendCell(row, text, header = false) {
  const cell = document.createElement(header ? "th" : "td");
  if (header) cell.scope = "row";
  cell.textContent = text;
  row.append(cell);
}

function updateComparison(data) {
  const body = document.querySelector("#comparison-rows");
  body.replaceChildren();
  for (const [model, label] of [["ols", "OLS"], ["ridge", "Ridge"]]) {
    const row = document.createElement("tr");
    appendCell(row, label, true);
    appendCell(row, String(data.degree));
    appendCell(row, model === "ols" ? "0" : displayLambda(data.lambda));
    appendCell(row, formatMetric(data.metrics[model].train_mse));
    appendCell(row, formatMetric(data.metrics[model].validation_mse));
    body.append(row);
  }
}

async function refresh() {
  const sequence = ++requestSequence;
  const controls = getDemoState();
  chartError.hidden = true;
  chartState.classList.remove("is-hidden");
  chartState.textContent = sequence === 1 ? "Loading the seeded example" : "Updating the fits";
  connectionState.dataset.state = "loading";
  connectionState.textContent = "Updating the example";

  try {
    const data = await fetchDemo(controls);
    if (sequence !== requestSequence) return;
    await Promise.all([
      renderFitChart(document.querySelector("#fit-chart"), data),
      renderCoefficientChart(document.querySelector("#coefficient-chart"), data.coefficients),
    ]);
    setMetric("ols-train-mse", data.metrics.ols.train_mse);
    setMetric("ols-validation-mse", data.metrics.ols.validation_mse);
    setMetric("ridge-train-mse", data.metrics.ridge.train_mse);
    setMetric("ridge-validation-mse", data.metrics.ridge.validation_mse);
    document.querySelector("#condition-number").textContent = data.condition_number_unscaled_train_design === null
      ? "Not finite"
      : formatMetric(data.condition_number_unscaled_train_design);
    updateComparison(data);
    chartState.classList.add("is-hidden");
    connectionState.dataset.state = "ready";
    connectionState.textContent = `Seed ${data.seed} · illustrative`;
  } catch (error) {
    if (sequence !== requestSequence) return;
    chartState.classList.add("is-hidden");
    chartError.hidden = false;
    connectionState.dataset.state = "error";
    connectionState.textContent = "Local API unavailable";
    document.querySelector("#fit-chart").setAttribute("aria-label", `The fit chart could not load: ${error.message}`);
  }
}

function scheduleRefresh() {
  updateControlLabels();
  window.clearTimeout(debounce);
  debounce = window.setTimeout(refresh, 120);
}

degreeInput.addEventListener("input", scheduleRefresh);
lambdaInput.addEventListener("input", scheduleRefresh);
zeroInput.addEventListener("change", scheduleRefresh);
document.querySelector("#retry").addEventListener("click", refresh);
document.querySelectorAll(".mobile-nav a").forEach((link) => {
  link.addEventListener("click", () => document.querySelector(".mobile-menu").removeAttribute("open"));
});

const sections = document.querySelectorAll(".section-anchor");
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      document.querySelectorAll(".primary-nav a").forEach((link) => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }
  }, { rootMargin: "-20% 0px -70% 0px" });
  sections.forEach((section) => observer.observe(section));
}

updateControlLabels();
initBikeExploratoryRanking();
refresh();
