export function getDemoState() {
  const noRegularization = document.querySelector("#no-regularization").checked;
  const exponent = Number(document.querySelector("#lambda").value);
  return {
    degree: Number(document.querySelector("#degree").value),
    lambda: noRegularization ? 0 : 10 ** exponent,
  };
}
