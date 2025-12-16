const talkBtn = document.getElementById("talkBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const player = document.getElementById("player");

const healthPill = document.getElementById("healthPill");
const sessionPill = document.getElementById("sessionPill");

const rightPanel = document.getElementById("rightPanel");
const toggleRight = document.getElementById("toggleRight");
const closeRight = document.getElementById("closeRight");
const backdrop = document.getElementById("backdrop");

const docsListEl = document.getElementById("docsList");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const docTitleEl = document.getElementById("docTitle");
const docMetaEl = document.getElementById("docMeta");
const docTextEl = document.getElementById("docText");

let sessionId = crypto.randomUUID();
sessionPill.textContent = `Session: ${sessionId.slice(0, 8)}`;

let mediaRecorder = null;
let chunks = [];

function setStatus(s) {
  statusEl.textContent = s;
}

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    healthPill.textContent = j.ok ? "Backend: OK" : "Backend: NOT OK";
  } catch {
    healthPill.textContent = "Backend: ni povezave";
  }
}
checkHealth();

function openRight() {
  rightPanel.classList.add("open");
  backdrop.classList.add("open");
}
function closeRightPanel() {
  rightPanel.classList.remove("open");
  backdrop.classList.remove("open");
}

toggleRight?.addEventListener("click", openRight);
closeRight?.addEventListener("click", closeRightPanel);
backdrop?.addEventListener("click", closeRightPanel);

async function startRecording() {
  setStatus("Snemam…");

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: "audio/webm" });
    await sendTurn(blob);
  };

  mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    setStatus("Pošiljam…");
    mediaRecorder.stop();
  }
}

async function sendTurn(audioBlob) {
  try {
    const fd = new FormData();
    fd.append("sessionId", sessionId);
    fd.append("audio", audioBlob, "audio.webm");

    const r = await fetch("/turn", { method: "POST", body: fd });
    const data = await r.json();

    if (!r.ok) {
      console.error(data);
      setStatus("Napaka (glej console)");
      return;
    }

    if (data.transcript) addBubble("Ti: " + data.transcript, "me");
    if (data.replyText) addBubble("Agent: " + data.replyText, "bot");

    if (data.replyAudioBase64) {
      player.src = data.replyAudioBase64;
      await player.play().catch(() => {});
    }

    setStatus("Pripravljen");
  } catch (e) {
    console.error(e);
    setStatus("Napaka (glej console)");
  }
}

talkBtn.addEventListener("mousedown", () => startRecording());
talkBtn.addEventListener("mouseup", () => stopRecording());
talkBtn.addEventListener("mouseleave", () => stopRecording());

talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); }, { passive: false });
talkBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); }, { passive: false });

resetBtn.addEventListener("click", async () => {
  try {
    await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId })
    });
  } catch {}

  sessionId = crypto.randomUUID();
  sessionPill.textContent = `Session: ${sessionId.slice(0, 8)}`;

  logEl.innerHTML = "";
  player.removeAttribute("src");
  player.load();
  setStatus("Pripravljen");
});

// ----------------- RAG docs UI -----------------

function renderDocsList(docs) {
  docsListEl.innerHTML = "";
  for (const d of docs) {
    const item = document.createElement("div");
    item.className = "docItem";
    item.innerHTML = `
      <div class="docItemTitle">${escapeHtml(d.title || d.id)}</div>
      <div class="docItemPreview">${escapeHtml(d.preview || "")}</div>
    `;
    item.addEventListener("click", () => loadDoc(d.id));
    docsListEl.appendChild(item);
  }
}

async function loadDocs() {
  try {
    const r = await fetch("/rag/docs");
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || "Failed");
    renderDocsList(j.docs || []);
  } catch (e) {
    console.error(e);
    docsListEl.innerHTML = `<div class="docsHint">Dokumentov ni mogoče naložiti.</div>`;
  }
}

async function loadDoc(id) {
  try {
    const r = await fetch(`/rag/docs/${encodeURIComponent(id)}`);
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || "Failed");
    docTitleEl.textContent = j.title || j.id;
    docMetaEl.textContent = `ID: ${j.id}`;
    docTextEl.textContent = j.text || "";
  } catch (e) {
    console.error(e);
    docTitleEl.textContent = "Napaka";
    docMetaEl.textContent = "";
    docTextEl.textContent = "Dokumenta ni mogoče naložiti.";
  }
}

refreshDocsBtn.addEventListener("click", loadDocs);
loadDocs();

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
