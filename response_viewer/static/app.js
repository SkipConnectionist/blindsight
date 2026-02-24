(function () {
  const dataEl = document.getElementById("seedData");
  const cardsEl = document.getElementById("cards");
  const filterEl = document.getElementById("filterSelect");
  const summaryEl = document.getElementById("summary");
  const matrixEl = document.getElementById("matrix");

  const payload = JSON.parse(dataEl.textContent || "{}");
  const allRecords = Array.isArray(payload.records) ? payload.records : [];
  const metrics = payload.global_metrics || {};

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function heatColor(value, maxValue) {
    if (maxValue <= 0) return "rgb(243, 244, 246)";
    const t = clamp(value / maxValue, 0, 1);
    const r = Math.round(243 - t * 191);
    const g = Math.round(244 - t * 52);
    const b = Math.round(246 - t * 62);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function renderConfusionMatrix() {
    if (!matrixEl) return;

    const tp = Number(metrics.true_positives ?? 0);
    const tn = Number(metrics.true_negatives ?? 0);
    const fp = Number(metrics.false_positives ?? 0);
    const fn = Number(metrics.false_negatives ?? 0);
    const maxCell = Math.max(tp, tn, fp, fn, 1);

    const cells = [
      { label: "TN", value: tn, x: 0, y: 0 },
      { label: "FP", value: fp, x: 1, y: 0 },
      { label: "FN", value: fn, x: 0, y: 1 },
      { label: "TP", value: tp, x: 1, y: 1 },
    ];

    const cell = 68;
    const gap = 4;
    const leftPad = 62;
    const topPad = 26;
    const rightPad = 8;
    const bottomPad = 24;
    const gridW = cell * 2 + gap;
    const gridH = cell * 2 + gap;
    const width = leftPad + gridW + rightPad;
    const height = topPad + gridH + bottomPad;

    const parts = [];
    parts.push(`<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Confusion matrix">`);
    parts.push('<rect x="0" y="0" width="100%" height="100%" fill="white"/>');

    parts.push(`<text x="${leftPad + gridW / 2}" y="14" text-anchor="middle" font-size="12" font-weight="700" fill="#111827">Predicted</text>`);
    parts.push(`<text x="${leftPad + cell / 2}" y="24" text-anchor="middle" font-size="10" fill="#374151">No</text>`);
    parts.push(`<text x="${leftPad + cell + gap + cell / 2}" y="24" text-anchor="middle" font-size="10" fill="#374151">Yes</text>`);

    parts.push(`<text x="25" y="${topPad + gridH / 2}" text-anchor="middle" font-size="12" font-weight="700" fill="#111827" transform="rotate(-90 25 ${topPad + gridH / 2})">Actual</text>`);
    parts.push(`<text x="${leftPad - 8}" y="${topPad + cell / 2 + 3}" text-anchor="end" font-size="10" fill="#374151">No</text>`);
    parts.push(`<text x="${leftPad - 8}" y="${topPad + cell + gap + cell / 2 + 3}" text-anchor="end" font-size="10" fill="#374151">Yes</text>`);

    for (const item of cells) {
      const x = leftPad + item.x * (cell + gap);
      const y = topPad + item.y * (cell + gap);
      const color = heatColor(item.value, maxCell);
      const textColor = "#111827";

      parts.push(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="6" ry="6" fill="${color}" stroke="#d1d5db"/>`);
      parts.push(`<text x="${x + cell / 2}" y="${y + 30}" text-anchor="middle" font-size="11" font-weight="700" fill="${textColor}">${item.label}</text>`);
      parts.push(`<text x="${x + cell / 2}" y="${y + 49}" text-anchor="middle" font-size="16" font-weight="700" fill="${textColor}">${item.value}</text>`);
    }

    parts.push(`<text x="${leftPad}" y="${height - 5}" font-size="10" fill="#6b7280">Lighter = lower count, darker = higher count</text>`);
    parts.push("</svg>");
    matrixEl.innerHTML = parts.join("");
  }

  function textForPrediction(value) {
    if (value === true) return "yes";
    if (value === false) return "no";
    return "unparsable";
  }

  function cardClass(record) {
    if (record.predicted_intersects === null) return "unknown";
    return record.correct ? "correct" : "incorrect";
  }

  function statusText(record) {
    const expected = record.expected_intersects ? "yes" : "no";
    const predicted = textForPrediction(record.predicted_intersects);

    if (record.predicted_intersects === null) {
      return `UNKNOWN: expected ${expected}, predicted ${predicted}`;
    }
    return record.correct
      ? `CORRECT: expected ${expected}, predicted ${predicted}`
      : `INCORRECT: expected ${expected}, predicted ${predicted}`;
  }

  function filteredRecords() {
    const mode = filterEl.value;
    if (mode === "correct") {
      return allRecords.filter((r) => r.correct === true);
    }
    if (mode === "incorrect") {
      return allRecords.filter((r) => r.correct === false);
    }
    return allRecords;
  }

  function render() {
    const rows = filteredRecords();
    cardsEl.innerHTML = "";

    for (const record of rows) {
      const card = document.createElement("article");
      card.className = `card ${cardClass(record)}`;

      card.innerHTML = `
        <img src="${record.image_url}" alt="${record.image_name}" loading="lazy" />
        <div class="content">
          <div class="filename">${record.image_name}</div>
          <div class="status">${statusText(record)}</div>
          <pre class="response">${escapeHtml(record.response_text)}</pre>
        </div>
      `;

      cardsEl.appendChild(card);
    }

    const total = allRecords.length;
    const shown = rows.length;
    const correct = allRecords.filter((r) => r.correct).length;
    const incorrect = allRecords.filter((r) => !r.correct).length;
    const truePositives = Number(metrics.true_positives ?? 0);
    const trueNegatives = Number(metrics.true_negatives ?? 0);
    const falsePositives = Number(metrics.false_positives ?? 0);
    const falseNegatives = Number(metrics.false_negatives ?? 0);
    const precision = Number(metrics.precision ?? 0);
    const recall = Number(metrics.recall ?? 0);

    summaryEl.textContent = `${shown}/${total} shown | correct: ${correct} | incorrect: ${incorrect} | TP: ${truePositives} | TN: ${trueNegatives} | FP: ${falsePositives} | FN: ${falseNegatives} | precision: ${precision.toFixed(3)} | recall: ${recall.toFixed(3)}`;
  }

  function escapeHtml(text) {
    return text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  filterEl.addEventListener("change", render);
  renderConfusionMatrix();
  render();
})();
