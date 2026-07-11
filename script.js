/* ================================================================
   script.js — StockSense Frontend Logic
   Connects to Flask backend at http://localhost:5000
   ================================================================ */

const BACKEND = "http://localhost:5000";

const STOCK_NAMES = {
  BBCA: "Bank Central Asia",
  BBRI: "Bank Rakyat Indonesia",
  BBNI: "Bank Negara Indonesia",
  BMRI: "Bank Mandiri",
};

/* ── State ─────────────────────────────────────────────────── */
let selectedStock = "BBCA";
let ohlcChart = null;

/* ── DOM refs ──────────────────────────────────────────────── */
const form = document.getElementById("priceForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = analyzeBtn.querySelector(".btn-text");
const btnIcon = analyzeBtn.querySelector(".btn-icon");
const btnSpinner = analyzeBtn.querySelector(".btn-spinner");
const validationMsg = document.getElementById("validationMsg");

const resultIdle = document.getElementById("resultIdle");
const resultCard = document.getElementById("resultCard");
const resultError = document.getElementById("resultError");

/* ================================================================
   STOCK SELECTOR
   ================================================================ */
document.querySelectorAll(".stock-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".stock-btn")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedStock = btn.dataset.stock;

    /* Reset result area when switching stock */
    showIdle();
    hideValidation();
  });
});

/* ================================================================
   FORM SUBMIT
   ================================================================ */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const o = parseFloat(document.getElementById("open").value);
  const h = parseFloat(document.getElementById("high").value);
  const l = parseFloat(document.getElementById("low").value);
  const c = parseFloat(document.getElementById("close").value);
  const date = document.getElementById("date").value;

  /* ── Client-side validation ──────────────────────────────── */
  if ([o, h, l, c].some(isNaN) || [o, h, l, c].some((v) => v <= 0)) {
    showValidation("Semua harga harus diisi dan bernilai > 0.");
    return;
  }
  if (h < l) {
    showValidation("High harus lebih besar atau sama dengan Low.");
    return;
  }
  if (h < o || h < c) {
    showValidation("High harus ≥ Open dan Close.");
    return;
  }
  if (l > o || l > c) {
    showValidation("Low harus ≤ Open dan Close.");
    return;
  }
  hideValidation();

  /* ── Loading state ───────────────────────────────────────── */
  setLoading(true);
  hideAll();

  /* ── Build FormData — field names harus cocok dengan Flask ── */
  const fd = new FormData();
  fd.append("ticker", selectedStock);
  fd.append("date", date);
  fd.append("open", o);
  fd.append("high", h);
  fd.append("low", l);
  fd.append("close", c);

  try {
    const resp = await fetch(`${BACKEND}/predict`, {
      method: "POST",
      body: fd,
      /* Jangan set Content-Type — biarkan browser set boundary otomatis */
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || `Server error ${resp.status}`);
      return;
    }

    renderResult(data, { o, h, l, c });
    drawChart(o, h, l, c, data.direction);
  } catch (err) {
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      showError(
        "Tidak dapat terhubung ke backend. Pastikan Flask berjalan: python app.py",
      );
    } else {
      showError(`Error: ${err.message}`);
    }
  } finally {
    setLoading(false);
  }
});

/* ================================================================
   RENDER RESULT
   ================================================================ */
function renderResult(data, prices) {
  const isUp = data.direction === "UP";

  /* Show result card, hide others */
  resultIdle.classList.add("hidden");
  resultError.classList.add("hidden");
  resultCard.classList.remove("hidden");

  /* Stock header */
  document.getElementById("resTicker").textContent = selectedStock;
  document.getElementById("resName").textContent =
    STOCK_NAMES[selectedStock] || selectedStock;
  document.getElementById("resDate").textContent = formatDate(new Date());

  /* Direction arrow */
  const predDirection = document.getElementById("predDirection");
  const predArrow = document.getElementById("predArrow");
  const predLabel = document.getElementById("predLabel");

  predDirection.className =
    "pred-direction " + (isUp ? "pred-up" : "pred-down");
  predLabel.textContent = isUp ? "UP" : "DOWN";

  predArrow.innerHTML = isUp
    ? `<svg width="48" height="48" viewBox="0 0 48 48" fill="none">
         <circle cx="24" cy="24" r="22" fill="${isUp ? "rgba(0,229,160,0.12)" : "rgba(255,77,109,0.12)"}"/>
         <path d="M24 32 L24 16 M16 24 L24 16 L32 24"
               stroke="${isUp ? "#00E5A0" : "#FF4D6D"}"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>`
    : `<svg width="48" height="48" viewBox="0 0 48 48" fill="none">
         <circle cx="24" cy="24" r="22" fill="rgba(255,77,109,0.12)"/>
         <path d="M24 16 L24 32 M16 24 L24 32 L32 24"
               stroke="#FF4D6D"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>`;

  /* Confidence — gunakan nilai asli dari model (desimal nyata) */
  const conf = parseFloat(data.confidence) || 0;

  document.getElementById("confValue").textContent = conf.toFixed(2) + "%";

  const fill = document.getElementById("confBarFill");
  fill.style.width = Math.min(conf, 100) + "%";
  fill.style.background = isUp ? "#00E5A0" : "#FF4D6D";

  /* OHLC summary */
  const fmt = (v) => "Rp " + Math.round(v).toLocaleString("id-ID");
  document.getElementById("sumOpen").textContent = fmt(prices.o);
  document.getElementById("sumHigh").textContent = fmt(prices.h);
  document.getElementById("sumLow").textContent = fmt(prices.l);
  document.getElementById("sumClose").textContent = fmt(prices.c);
}

/* ================================================================
   CHART — simulated OHLC candle + probability bars
   ================================================================ */
function drawChart(o, h, l, c, direction) {
  const canvas = document.getElementById("ohlcChart");
  if (!canvas) return;

  if (ohlcChart) {
    ohlcChart.destroy();
    ohlcChart = null;
  }

  const isUp = direction === "UP";
  const barColor = isUp ? "#00E5A0" : "#FF4D6D";
  const midColor = "rgba(255,255,255,0.08)";

  /* Build a simple bar chart showing OHLC as a range */
  const ctx = canvas.getContext("2d");

  /* Normalise values for display (0-100 scale relative to range) */
  const range = h - l || 1;
  const oNorm = (((o - l) / range) * 80 + 10).toFixed(1);
  const hNorm = 90;
  const lNorm = 10;
  const cNorm = (((c - l) / range) * 80 + 10).toFixed(1);

  ohlcChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["LOW", "OPEN", "CLOSE", "HIGH"],
      datasets: [
        {
          data: [lNorm, oNorm, cNorm, hNorm],
          backgroundColor: [midColor, midColor, barColor, midColor],
          borderColor: [midColor, midColor, barColor, midColor],
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const vals = { LOW: l, OPEN: o, CLOSE: c, HIGH: h };
              const key = ctx.label;
              return ` Rp ${Math.round(vals[key]).toLocaleString("id-ID")}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#4A6580", font: { family: "Space Mono", size: 10 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: {
            color: "#4A6580",
            font: { family: "Space Mono", size: 10 },
            callback: () => "" /* Hide Y axis labels — relative scale */,
          },
          grid: { color: "rgba(255,255,255,0.04)" },
          min: 0,
          max: 100,
        },
      },
    },
  });
}

/* ================================================================
   MODEL STATUS POLLING
   ================================================================ */
async function fetchStatus() {
  try {
    const resp = await fetch(`${BACKEND}/status`);
    if (!resp.ok) throw new Error("not ok");
    const data = await resp.json();
    updateStatusDots(data);
  } catch {
    /* Backend offline — semua dot merah */
    updateStatusDots({ BBCA: false, BBRI: false, BBNI: false, BMRI: false });
  }
}

function updateStatusDots(statusMap) {
  document.querySelectorAll(".status-item").forEach((item) => {
    const ticker = item.dataset.stock;
    const ready = statusMap[ticker] === true;

    item.classList.remove("loading", "ready", "offline");
    item.classList.add(ready ? "ready" : "offline");

    const dot = item.querySelector(".s-dot");
    if (dot) {
      dot.style.background = ready ? "#00E5A0" : "#FF4D6D";
    }
  });
}

document.getElementById("refreshStatus").addEventListener("click", fetchStatus);

/* Poll on load, then every 10 seconds */
fetchStatus();
setInterval(fetchStatus, 10_000);

/* ================================================================
   UI HELPERS
   ================================================================ */
function setLoading(on) {
  analyzeBtn.disabled = on;
  btnText.classList.toggle("hidden", on);
  btnIcon.classList.toggle("hidden", on);
  btnSpinner.classList.toggle("hidden", !on);
}

function showIdle() {
  resultIdle.classList.remove("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.add("hidden");
}

function hideAll() {
  resultIdle.classList.add("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.add("hidden");
}

function showError(msg) {
  resultIdle.classList.add("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.remove("hidden");
  document.getElementById("errorMsg").textContent = msg;
}

function showValidation(msg) {
  validationMsg.textContent = msg;
  validationMsg.classList.remove("hidden");
}

function hideValidation() {
  validationMsg.classList.add("hidden");
}

function formatDate(d) {
  return d.toLocaleDateString("id-ID", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}
