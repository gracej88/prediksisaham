/* ================================================================
   Konfigurasi Backend
   ================================================================ */

// Alamat Flask backend yang akan menerima request dari frontend

const BACKEND = "http://localhost:5000";

/* ================================================================
   Daftar Saham
   ================================================================ */

// Menyimpan nama lengkap masing-masing kode saham
// sehingga dapat ditampilkan pada halaman hasil prediksi.
const STOCK_NAMES = {
  BBCA: "Bank Central Asia",
  BBRI: "Bank Rakyat Indonesia",
  BBNI: "Bank Negara Indonesia",
  BMRI: "Bank Mandiri",
};

/* ================================================================
   Variabel Global
   ================================================================ */

// Menyimpan saham yang sedang dipilih pengguna
let selectedStock = "BBCA";

// Menyimpan objek grafik Chart.js
// agar dapat diperbarui ketika prediksi baru dilakukan.
let ohlcChart = null;

/* ================================================================
   Referensi Elemen HTML
   ================================================================ */

// Mengambil seluruh elemen HTML yang akan digunakan
// selama proses interaksi dengan pengguna.
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
   Pemilihan Saham
   ================================================================ */

/*
Fungsi:
1. Mendeteksi ketika pengguna memilih salah satu tombol saham.
2. Menghapus status aktif (active) dari semua tombol saham.
3. Memberikan status aktif pada tombol yang dipilih.
4. Menyimpan kode saham yang dipilih ke variabel selectedStock.
5. Menghapus hasil prediksi sebelumnya agar pengguna melakukan analisis baru.
6. Menyembunyikan pesan validasi jika sebelumnya ada kesalahan input.
*/

// Mengambil seluruh tombol saham yang memiliki class "stock-btn"
document.querySelectorAll(".stock-btn").forEach((btn) => {
  // Menjalankan fungsi ketika tombol diklik
  btn.addEventListener("click", () => {
    // Menghapus class "active" dari seluruh tombol saham
    // sehingga hanya ada satu tombol yang aktif
    document
      .querySelectorAll(".stock-btn")
      .forEach((b) => b.classList.remove("active"));

    // Memberikan class "active" pada tombol yang dipilih
    btn.classList.add("active");

    // Menyimpan kode saham yang dipilih (BBCA, BBRI, BBNI, atau BMRI)
    // ke dalam variabel selectedStock
    selectedStock = btn.dataset.stock;

    // Mengembalikan tampilan hasil ke kondisi awal (belum ada prediksi)
    showIdle();
    // Menyembunyikan pesan validasi apabila sebelumnya muncul
    hideValidation();
  });
});

/* ================================================================
   FORM SUBMIT
   ================================================================ */

/*
Fungsi:
Mengambil data input pengguna, melakukan validasi, mengirim data
ke backend Flask, kemudian menampilkan hasil prediksi.
*/
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const o = parseFloat(document.getElementById("open").value);
  const h = parseFloat(document.getElementById("high").value);
  const l = parseFloat(document.getElementById("low").value);
  const c = parseFloat(document.getElementById("close").value);
  const date = document.getElementById("date").value;

  /* Validasi input pengguna */
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

  /* Mengaktifkan loading dan menyembunyikan hasil sebelumnya */
  setLoading(true);
  hideAll();

  /* Menyiapkan data yang akan dikirim ke Flask */
  const fd = new FormData();
  fd.append("ticker", selectedStock);
  fd.append("date", date);
  fd.append("open", o);
  fd.append("high", h);
  fd.append("low", l);
  fd.append("close", c);

  /* Mengirim request prediksi ke backend */
  try {
    const resp = await fetch(`${BACKEND}/predict`, {
      method: "POST",
      body: fd,
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || `Server error ${resp.status}`);
      return;
    }

    /* Menampilkan hasil prediksi dan grafik */
    renderResult(data, { o, h, l, c });
    drawChart(o, h, l, c, data.direction);
  } catch (err) {
    /* Menampilkan pesan jika backend tidak dapat diakses */
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      showError(
        "Tidak dapat terhubung ke backend. Pastikan Flask berjalan: python app.py",
      );
    } else {
      showError(`Error: ${err.message}`);
    }
  } finally {
    /* Mengembalikan tombol ke kondisi normal */
    setLoading(false);
  }
});

/* ================================================================ RENDER RESULT ================================================================ */
/* Fungsi: Menampilkan hasil prediksi yang diterima dari backend, meliputi arah pergerakan saham, 
confidence score, serta ringkasan data OHLC yang diinput pengguna. */
function renderResult(data, prices) {
  const isUp = data.direction === "UP";

  /* Menampilkan area hasil prediksi */
  resultIdle.classList.add("hidden");
  resultError.classList.add("hidden");
  resultCard.classList.remove("hidden");

  /* Menampilkan informasi saham */
  document.getElementById("resTicker").textContent = selectedStock;
  document.getElementById("resName").textContent =
    STOCK_NAMES[selectedStock] || selectedStock;
  document.getElementById("resDate").textContent = formatDate(new Date());

  /* Menampilkan arah prediksi */
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

  /* Menampilkan nilai confidence hasil prediksi */
  const conf = parseFloat(data.confidence) || 0;

  document.getElementById("confValue").textContent = conf.toFixed(2) + "%";

  const fill = document.getElementById("confBarFill");
  fill.style.width = Math.min(conf, 100) + "%";
  fill.style.background = isUp ? "#00E5A0" : "#FF4D6D";

  /* Menampilkan ringkasan harga Open, High, Low, dan Close */
  const fmt = (v) => "Rp " + Math.round(v).toLocaleString("id-ID");
  document.getElementById("sumOpen").textContent = fmt(prices.o);
  document.getElementById("sumHigh").textContent = fmt(prices.h);
  document.getElementById("sumLow").textContent = fmt(prices.l);
  document.getElementById("sumClose").textContent = fmt(prices.c);
}

/* ================================================================ CHART — Visualisasi OHLC ================================================================ */
/* Fungsi: Membuat grafik OHLC berdasarkan data harga yang dimasukkan pengguna. 
Grafik diperbarui setiap kali proses prediksi selesai menggunakan library Chart.js. */
function drawChart(o, h, l, c, direction) {
  const canvas = document.getElementById("ohlcChart");
  if (!canvas) return;

  /* Menghapus grafik sebelumnya agar tidak terjadi duplikasi */
  if (ohlcChart) {
    ohlcChart.destroy();
    ohlcChart = null;
  }

  const isUp = direction === "UP";
  const barColor = isUp ? "#00E5A0" : "#FF4D6D";
  const midColor = "rgba(255,255,255,0.08)";

  const ctx = canvas.getContext("2d");

  /* Normalisasi nilai OHLC ke skala 0–100 agar mudah divisualisasikan */
  const range = h - l || 1;
  const oNorm = (((o - l) / range) * 80 + 10).toFixed(1);
  const hNorm = 90;
  const lNorm = 10;
  const cNorm = (((c - l) / range) * 80 + 10).toFixed(1);

  /* Membuat grafik batang menggunakan Chart.js */
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
        /* Menampilkan nilai harga asli saat kursor diarahkan ke grafik */
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
            /* Menyembunyikan label sumbu Y karena hanya menggunakan skala relatif */
            callback: () => "",
          },
          grid: { color: "rgba(255,255,255,0.04)" },
          min: 0,
          max: 100,
        },
      },
    },
  });
}

/* =============== MODEL STATUS POLLING ================== */
/* Fungsi: Mengecek status model Random Forest pada backend Flask. 
Informasi ini digunakan untuk menampilkan apakah model masing-masing saham sudah siap digunakan. */
async function fetchStatus() {
  try {
    const resp = await fetch(`${BACKEND}/status`);
    if (!resp.ok) throw new Error("not ok");
    const data = await resp.json();
    // Memperbarui indikator status model pada halaman
    updateStatusDots(data);
  } catch {
    // Jika backend tidak dapat diakses, seluruh status menjadi offline
    updateStatusDots({ BBCA: false, BBRI: false, BBNI: false, BMRI: false });
  }
}

/* Fungsi: Memperbarui indikator status setiap model berdasarkan 
data yang diterima dari backend. */
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

/* Tombol untuk memperbarui status model secara manual */
document.getElementById("refreshStatus").addEventListener("click", fetchStatus);

/* Melakukan pengecekan status saat halaman dibuka dan setiap 10 detik sekali */
fetchStatus();
setInterval(fetchStatus, 10_000);

/* ================================================================
   UI HELPERS
   ================================================================ */
/* Fungsi-fungsi berikut digunakan untuk mengatur tampilan antarmuka pengguna selama proses analisis. */

/* Mengubah tampilan tombol Analyze menjadi loading */
function setLoading(on) {
  analyzeBtn.disabled = on;
  btnText.classList.toggle("hidden", on);
  btnIcon.classList.toggle("hidden", on);
  btnSpinner.classList.toggle("hidden", !on);
}
/* Menampilkan tampilan awal sebelum hasil prediksi muncul */
function showIdle() {
  resultIdle.classList.remove("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.add("hidden");
}
/* Menyembunyikan seluruh area hasil */
function hideAll() {
  resultIdle.classList.add("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.add("hidden");
}
/* Menampilkan pesan kesalahan */
function showError(msg) {
  resultIdle.classList.add("hidden");
  resultCard.classList.add("hidden");
  resultError.classList.remove("hidden");
  document.getElementById("errorMsg").textContent = msg;
}
/* Menampilkan pesan validasi input */
function showValidation(msg) {
  validationMsg.textContent = msg;
  validationMsg.classList.remove("hidden");
}
/* Menyembunyikan pesan validasi */
function hideValidation() {
  validationMsg.classList.add("hidden");
}

/* ====================== FORMAT TANGGAL =============== */
/* Fungsi: Mengubah format tanggal menjadi format Indonesia (dd mmm yyyy) 
untuk ditampilkan pada halaman hasil. */

function formatDate(d) {
  return d.toLocaleDateString("id-ID", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}
