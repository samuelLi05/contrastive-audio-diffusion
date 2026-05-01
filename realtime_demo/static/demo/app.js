const form = document.getElementById("demo-form");
const button = document.getElementById("run-button");
const message = document.getElementById("message");
const referenceAudio = document.getElementById("reference-audio");
const noisyInputAudio = document.getElementById("noisy-input-audio");
const generatedAudio = document.getElementById("generated-audio");
const classNamesInput = document.getElementById("class-names-input");
const referenceClassSelect = document.getElementById("reference-class-select");
const targetClassSelect = document.getElementById("target-class-select");
const canvas = document.getElementById("benchmark-chart");
const ctx = canvas.getContext("2d");

function csrfToken() {
  return document.querySelector("[name=csrfmiddlewaretoken]").value;
}

function drawChart(points) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = 44;
  const width = canvas.width - pad * 2;
  const height = canvas.height - pad * 2;
  const maxSteps = Math.max(...points.map((p) => p.steps), 1);
  const maxTime = Math.max(...points.map((p) => p.elapsed_ms), 1);

  ctx.strokeStyle = "#ccd5d9";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + height);
  ctx.lineTo(pad + width, pad + height);
  ctx.stroke();

  ctx.fillStyle = "#5d6b74";
  ctx.font = "13px system-ui";
  ctx.fillText("elapsed ms", pad, 22);
  ctx.fillText("steps", pad + width - 28, pad + height + 30);

  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = pad + (point.steps / maxSteps) * width;
    const y = pad + height - (point.elapsed_ms / maxTime) * height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  points.forEach((point) => {
    const x = pad + (point.steps / maxSteps) * width;
    const y = pad + height - (point.elapsed_ms / maxTime) * height;
    ctx.fillStyle = "#0f766e";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#172026";
    ctx.fillText(`${point.steps} / ${point.elapsed_ms}ms`, x + 8, y - 8);
  });
}

function classNamesFromInput() {
  return classNamesInput.value
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name.length > 0);
}

function repopulateClassSelect(select, fallbackName) {
  const previous = select.value || fallbackName;
  const names = classNamesFromInput();
  select.innerHTML = "";
  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === previous || (!names.includes(previous) && name === fallbackName)) {
      option.selected = true;
    }
    select.appendChild(option);
  });
}

function syncClassSelects() {
  repopulateClassSelect(referenceClassSelect, "bass");
  repopulateClassSelect(targetClassSelect, "guitar");
}

function setImage(id, url) {
  const image = document.getElementById(id);
  const figure = image.closest("figure");
  if (url) {
    image.src = url;
    figure.classList.add("has-image");
  } else {
    image.removeAttribute("src");
    figure.classList.remove("has-image");
  }
}

function fillRows(id, rows, renderRow) {
  const body = document.getElementById(id);
  body.innerHTML = "";
  if (!rows || rows.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="3">-</td>';
    body.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = renderRow(item);
    body.appendChild(row);
  });
}

function updateReferenceMetrics(metrics) {
  const refMetrics = metrics.reference_metrics;
  if (!refMetrics || !refMetrics.available) {
    document.getElementById("metrics-message").textContent =
      refMetrics?.message || "Reference-based metrics are unavailable for this run.";
    ["snr-noisy", "snr-output", "snr-gain", "mse-output"].forEach((id) => {
      document.getElementById(id).textContent = "-";
    });
    setImage("psnr-heatmap", "");
    setImage("segment-snr", "");
    setImage("mel-difference", "");
    fillRows("worst-tf-regions", [], () => "");
    fillRows("worst-segments", [], () => "");
    return;
  }

  document.getElementById("metrics-message").textContent =
    "Reference metrics compare the generated output against the clean reference and noisy model input.";
  document.getElementById("snr-noisy").textContent = `${refMetrics.snr_noisy_db} dB`;
  document.getElementById("snr-output").textContent = `${refMetrics.snr_output_db} dB`;
  document.getElementById("snr-gain").textContent = `${refMetrics.snr_improvement_db} dB`;
  document.getElementById("mse-output").textContent = refMetrics.mse_output;
  setImage("psnr-heatmap", refMetrics.psnr_heatmap_url);
  setImage("segment-snr", refMetrics.segment_snr_url);
  setImage("mel-difference", refMetrics.mel_difference_url);
  fillRows(
    "worst-tf-regions",
    refMetrics.worst_tf_regions,
    (item) => `<td>${item.time_sec}s</td><td>${item.freq_hz} Hz</td><td>${item.psnr_db} dB</td>`,
  );
  fillRows(
    "worst-segments",
    refMetrics.worst_segments,
    (item) =>
      `<td>${item.start_sec}-${item.end_sec}s</td><td>${item.improvement_db} dB</td><td>${item.output_snr_db} dB</td>`,
  );
}

classNamesInput.addEventListener("change", syncClassSelects);
classNamesInput.addEventListener("blur", syncClassSelects);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  message.textContent = "Generating audio and collecting benchmark timings...";

  try {
    const response = await fetch("/api/generate/", {
      method: "POST",
      headers: {"X-CSRFToken": csrfToken()},
      body: new FormData(form),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Generation failed.");
    }

    referenceAudio.src = payload.reference_audio_url;
    noisyInputAudio.src = payload.noisy_input_audio_url;
    generatedAudio.src = payload.generated_audio_url;
    drawChart(payload.benchmark);

    document.getElementById("duration").textContent = `${payload.duration_sec}s`;
    document.getElementById("reference-rms").textContent = payload.metrics.reference_rms;
    document.getElementById("noisy-rms").textContent = payload.metrics.noisy_input_rms;
    document.getElementById("generated-rms").textContent = payload.metrics.generated_rms;
    document.getElementById("peak-memory").textContent =
      payload.metrics.peak_memory_mb === null ? "cpu" : `${payload.metrics.peak_memory_mb} MB`;
    updateReferenceMetrics(payload.metrics);

    const fastest = payload.benchmark[0];
    const finalRun = payload.benchmark[payload.benchmark.length - 1];
    message.textContent =
      `Generated ${payload.target_class} from ${payload.reference_label}. ` +
      `Fastest: ${fastest.elapsed_ms} ms, final: ${finalRun.elapsed_ms} ms, ` +
      `RTF ${finalRun.real_time_factor}.`;
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
