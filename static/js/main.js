// База API: тот же домен (nginx проксирует /upload и /result к FastAPI)
const API_BASE = "http://127.0.0.1:8000";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const noPreview = document.getElementById("noPreview");

// эти два не используем для показа (оставлены как есть, без удаления)
const resultImg = document.getElementById("resultImg");
const resultEmpty = document.getElementById("resultEmpty");

const processBtn = document.getElementById("processBtn");
const statusEl = document.getElementById("status");
const downloadBtn = document.getElementById("downloadBtn");
const copyLinkBtn = document.getElementById("copyLinkBtn");

let currentFile = null;

// интерактивный выбор стиля
const styleInput = document.getElementById("styleSelect");
const stylePicker = document.getElementById("stylePicker");
const styleButtons = stylePicker.querySelectorAll("[data-style]");

// отметить активный
function activate(btn) {
  styleButtons.forEach((b) => {
    b.classList.remove("scale-105", "shadow-xl", "shadow-black", "is-anim");
    b.setAttribute("aria-pressed", "false");
  });

  btn.classList.add("scale-105", "shadow-xl", "shadow-black");
  btn.setAttribute("aria-pressed", "true");

  // перезапуск анимации вспышки на кнопке
  btn.classList.remove("is-anim");
  void btn.offsetWidth;
  btn.classList.add("is-anim");
}

styleButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    styleInput.value = btn.dataset.style; // совместимо с остальным кодом
    activate(btn);
  });
});

// по умолчанию — comic
activate(styleButtons[0]);

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function showPreview(file) {
  const url = URL.createObjectURL(file);
  // лёгкий fade-in и для локального превью
  preview.classList.remove("fade-in");
  void preview.offsetWidth;
  preview.src = url;
  preview.classList.add("fade-in");

  preview.classList.remove("hidden");
  noPreview.classList.add("hidden");
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("border-indigo-500");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("border-indigo-500")
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("border-indigo-500");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    alert("Only JPEG/PNG/WEBP are allowed.");
    return;
  }
  currentFile = file;
  showPreview(file);
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    alert("Only JPEG/PNG/WEBP are allowed.");
    return;
  }
  currentFile = file;
  showPreview(file);
});

async function processImage() {
  const style = document.getElementById("styleSelect").value;
  if (!currentFile) {
    alert("Choose an image first.");
    return;
  }

  const form = new FormData();
  form.append("file", currentFile);

  processBtn.disabled = true;
  setStatus("Processing…");

  try {
    const res = await fetch(
      `${API_BASE}/upload?style=${encodeURIComponent(style)}`,
      {
        method: "POST",
        body: form,
      }
    );
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const resultUrl = `${API_BASE}${data.result_url}`; // /result/{uid}

    // === ПОКАЗ РЕЗУЛЬТАТА ПОВЕРХ ОРИГИНАЛА + FADE-IN ===
    preview.classList.remove("fade-in");
    void preview.offsetWidth; // force reflow для перезапуска анимации
    preview.src = `${resultUrl}?t=${Date.now()}`; // cache-bust
    preview.classList.add("fade-in");

    // Кнопки "Скачать" и "Copy Link" — как было
    downloadBtn.href = resultUrl;
    downloadBtn.classList.remove("hidden");
    downloadBtn.setAttribute("download", `${data.id}.jpg`);

    copyLinkBtn.classList.remove("hidden");
    copyLinkBtn.onclick = async () => {
      await navigator.clipboard.writeText(resultUrl);
      copyLinkBtn.textContent = "Copied!";
      setTimeout(() => (copyLinkBtn.textContent = "Copy Link"), 1200);
    };

    setStatus("Done");
  } catch (err) {
    console.error(err);
    setStatus("Error");
    alert(`Upload failed: ${err.message}`);
  } finally {
    processBtn.disabled = false;
  }
}

processBtn.addEventListener("click", processImage);

// Лёгкая анимация появления результата — добавляем CSS через JS (чтобы HTML не менять)
const _style = document.createElement("style");
_style.textContent = `
  .fade-in { animation: fadeIn .3s ease-out forwards; }
  @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
`;
document.head.appendChild(_style);
