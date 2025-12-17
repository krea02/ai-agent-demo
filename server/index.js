import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import multer from "multer";
import { fileURLToPath } from "url";

import { loadDocs, retrieveDocs } from "./rag.js";
import { calculate_premium, normalizeCoverageLevel } from "./premium.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, "..", ".env") });

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));
const webDir = path.join(__dirname, "..", "web");

app.use(express.static(webDir));
app.get("/", (req, res) => res.sendFile(path.join(webDir, "index.html")));

const upload = multer({ storage: multer.memoryStorage() });

const sessions = new Map();

function getSession(sessionId) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, {
      messages: [
        {
          role: "system",
          content:
            "Ti si prijazen inbound AI voice agent zavarovalnice. " +
            "Odgovarjaj v slovenščini. Če je izjava nejasna, postavi eno kratko dodatno vprašanje. " +
            "Odgovori naj bodo kratki, praktični in razumljivi."
        }
      ],
      state: {
        premium: null,
        lastPremium: null,
        comparedLevels: new Set(),
        pendingComparison: null 
      }
    });
  }
  return sessions.get(sessionId);
}

const docsDir = path.join(__dirname, "..", "docs");

let docs = [];
try {
  docs = loadDocs(docsDir);
  console.log(`Loaded ${docs.length} docs from ${docsDir}`);
} catch (e) {
  console.warn(`Docs not loaded from ${docsDir}:`, e?.message || e);
  docs = [];
}

// ---------------- OpenAI helpers ----------------

async function sttTranscribe(audioBuffer, filename) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("Missing OPENAI_API_KEY in .env");

  const form = new FormData();
  form.set("file", new Blob([audioBuffer], { type: "audio/webm" }), filename || "audio.webm");
  form.set("model", "gpt-4o-mini-transcribe");
  form.set("language", "sl");

  const r = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form
  });

  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error(`STT failed (${r.status}): ${JSON.stringify(data) || "no body"}`);
  return (data?.text || "").trim();
}

function tailMessages(messages, maxPairs = 8) {
  const sys = messages.length ? [messages[0]] : [];
  const rest = messages.slice(1);
  const tail = rest.slice(Math.max(0, rest.length - maxPairs * 2));
  return [...sys, ...tail];
}

async function llmReply(messages) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("Missing OPENAI_API_KEY in .env");

  const trimmed = tailMessages(messages, 8);

  const r = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      input: trimmed.map((m) => ({ role: m.role, content: m.content })),
      text: { format: { type: "text" } }
    })
  });

  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error(`LLM failed (${r.status}): ${JSON.stringify(data) || "no body"}`);

  const out = (data.output || [])
    .flatMap((item) => item.content || [])
    .filter((c) => c.type === "output_text" && typeof c.text === "string")
    .map((c) => c.text)
    .join("\n")
    .trim();

  return out || "Se opravičujem — lahko ponoviš vprašanje?";
}

async function ttsSpeak(text) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("Missing OPENAI_API_KEY in .env");

  const r = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      input: text,
      format: "mp3"
    })
  });

  if (!r.ok) {
    const errText = await r.text().catch(() => "");
    throw new Error(`TTS failed (${r.status}): ${errText}`);
  }

  const arrayBuf = await r.arrayBuffer();
  return Buffer.from(arrayBuf);
}

async function respondWithTTS(res, transcript, replyText, extra = {}) {
  const mp3 = await ttsSpeak(replyText);
  const base64 = mp3.toString("base64");
  res.json({
    transcript,
    replyText,
    replyAudioBase64: `data:audio/mpeg;base64,${base64}`,
    ...extra
  });
}

// ---------------- parsing helpers ----------------

function cleanTextForNumber(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/\bzdeset\b/gu, "deset")
    .replace(/[^\p{L}\p{M}\d\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeWord(w) {
  return (w || "").toLowerCase().trim();
}

function isWordLike(word, target) {
  const w = normalizeWord(word);
  const t = normalizeWord(target);
  if (w === t) return true;
  if (w.startsWith(t.slice(0, 3)) && w.length >= 3) return true;
  const w2 = w.replace(/u/g, "v");
  const t2 = t.replace(/u/g, "v");
  if (w2 === t2) return true;
  return false;
}

function normYesNo(text) {
  return (text || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[^\p{L}\p{M}\d\s]/gu, " ") 
    .replace(/\s+/g, " ")
    .trim();
}
function firstToken(text) {
  return normYesNo(text).split(" ").filter(Boolean)[0] || "";
}
function isAffirmative(text) {
  const t = normYesNo(text);
  const ft = firstToken(text);
  if (ft === "da" || ft === "ja") return true;
  if (t.startsWith("seveda")) return true;
  if (t.startsWith("ok")) return true;
  if (t.startsWith("lahko")) return true;
  if (ft === "mhm" || ft === "mhmm") return true;
  if (ft === "valda") return true;
  return false;
}
function isNegative(text) {
  const t = normYesNo(text);
  const ft = firstToken(text);
  if (ft === "ne" || ft === "no") return true;
  if (t.startsWith("ne hvala")) return true;
  if (t.startsWith("ni treba")) return true;
  if (ft === "nic" || ft === "nič" || ft === "nc" || ft === "nč") return true;
  return false;
}

function parseSlNumberRobust(text) {
  const t = cleanTextForNumber(text);

  const digit = t.match(/\b(\d{1,4})\b/);
  if (digit) return Number(digit[1]);

  const words = t.split(" ").filter(Boolean);
  if (!words.length) return null;

  const ones = {
    "nič": 0, "nic": 0, "nula": 0,
    "ena": 1, "en": 1, "eno": 1,
    "dva": 2, "dve": 2,
    "tri": 3,
    "štiri": 4, "stiri": 4,
    "pet": 5,
    "šest": 6, "sest": 6,
    "sedem": 7,
    "osem": 8,
    "devet": 9
  };

  const teens = {
    "deset": 10,
    "enajst": 11,
    "dvanajst": 12,
    "trinajst": 13,
    "štirinajst": 14, "stirinajst": 14,
    "petnajst": 15,
    "šestnajst": 16, "sestnajst": 16,
    "sedemnajst": 17,
    "osemnajst": 18,
    "devetnajst": 19
  };

  const tens = {
    "dvajset": 20,
    "trideset": 30,
    "štirideset": 40, "stirideset": 40,
    "petdeset": 50, "petindeset": 50,
    "šestdeset": 60, "sestdeset": 60, "šezdeset": 60, "sezdeset": 60, "sestindeset": 60,
    "sedemdeset": 70, "sedemindeset": 70,
    "osemdeset": 80, "osemindeset": 80,
    "devetdeset": 90, "devetindeset": 90
  };

  const hundreds = [
    ["devetsto", 900],
    ["osemsto", 800],
    ["sedemsto", 700],
    ["šeststo", 600], ["seststo", 600],
    ["petsto", 500],
    ["štiristo", 400], ["stiristo", 400],
    ["tristo", 300],
    ["dvesto", 200],
    ["dve sto", 200], 
    ["sto", 100]
  ];

  function parseBelow100(s0) {
    const s = Array.isArray(s0) ? s0.join(" ") : s0;

    for (const [k, v] of Object.entries(tens)) {
      if (s.includes(k)) {
        const compact = s.replace(/\s+/g, "");
        for (const [oneK, oneV] of Object.entries(ones)) {
          if (compact.includes(`${oneK}in${k}`)) return v + oneV;
        }
        const m = s.match(new RegExp(`\\b${k}\\b(?:\\s+in\\s+|\\s+)?([\\p{L}\\p{M}]+)\\b`, "iu"));
        if (m) {
          const w = m[1].toLowerCase();
          if (ones[w] != null) return v + ones[w];
          if (isWordLike(w, "devet")) return v + 9;
        }
        return v;
      }
    }

    for (const [k, v] of Object.entries(teens)) if (s.includes(k)) return v;

    const ws = s.split(" ").filter(Boolean);
    for (let i = 0; i < ws.length - 1; i++) {
      const a = ws[i];
      const b = ws[i + 1];
      if (b === "deset") {
        if (ones[a] != null) return ones[a] * 10;
        if (isWordLike(a, "devet")) return 90;
        if (isWordLike(a, "osem")) return 80;
        if (isWordLike(a, "sedem")) return 70;
      }
    }

    for (const [k, v] of Object.entries(ones)) if (s.includes(k)) return v;
    if (ws.length === 1 && isWordLike(ws[0], "devet")) return 9;

    return null;
  }

  const joined = words.join(" ");

  for (const [hWord, hVal] of hundreds) {
    const pattern = new RegExp(`(^|\\s)${hWord}(\\s|$)`, "iu");
    if (pattern.test(joined)) {
      const parts = joined.split(new RegExp(`\\b${hWord}\\b`, "iu"));
      const after = (parts[1] || "").trim();
      if (!after) return hVal;

      const rest = parseBelow100(after);
      if (rest != null) return hVal + rest;
      return hVal;
    }
  }

  return parseBelow100(words);
}


function extractVehicleAge(text) {
  const t = cleanTextForNumber(text);

  let m = t.match(/\b(\d{1,2})\s*(let|leto|leti|letih|lit|lita|liti)\b/);
  if (m) return Number(m[1]);

  m = t.match(/star\w*\s*(\d{1,2})/);
  if (m) return Number(m[1]);

  const n = parseSlNumberRobust(t);
  if (n != null && n >= 0 && n <= 40) return n;

  return null;
}

function extractHorsepower(text) {
  const t = cleanTextForNumber(text);

  let m = t.match(/\b(\d{2,4})\s*(km|konj|konji|konjev|konjska)\b/);
  if (m) return Number(m[1]);

  m = t.match(/moč\w*\s*(\d{2,4})/);
  if (m) return Number(m[1]);

  const n = parseSlNumberRobust(t);
  if (n != null && n >= 20 && n <= 600) return n;

  return null;
}

function extractCoverageLevel(text, { wantsPremium = false } = {}) {
  const norm = normalizeCoverageLevel(text);
  if (norm) return norm;

  const t = (text || "").toLowerCase();
  if (t.includes("osnov")) return "osnovno";
  if (t.includes("delni")) return "delni_kasko";
  if (t.includes("polni")) return "polni_kasko";

  if (wantsPremium && t.includes("kasko")) return "polni_kasko";
  return null;
}

function mentionsCoverage(text) {
  const t = (text || "").toLowerCase();
  return t.includes("kasko") || t.includes("delni") || t.includes("polni") || t.includes("osnov");
}

function isProbablyNotCityWord(raw) {
  const t = (raw || "").toLowerCase().trim();
  if (t.includes("kasko") || t.includes("osnov")) return true;
  if (t === "da" || t === "ne" || t === "ja") return true;
  if (t.length < 3) return true;
  return false;
}

function extractCity(text) {
  const raw0 = (text || "").trim();
  const raw = raw0
    .replace(/[^\p{L}\p{M}\- ]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  const t = raw.toLowerCase();
  const compact = t.replace(/\s+/g, "");

  if (
    compact === "vseeno" ||
    compact === "vseno" ||
    compact === "karkoli" ||
    compact === "kjerkoli" ||
    t.includes("ni pomembno") ||
    t.includes("nima veze")
  ) return "Other";

  if (mentionsCoverage(raw) || extractCoverageLevel(raw, { wantsPremium: false })) return null;
  if (isProbablyNotCityWord(raw)) return null;

  const maybeNum = parseSlNumberRobust(t);
  if (maybeNum != null) return null;

  const cityNormMap = {
    "kopru": "Koper", "koper": "Koper",
    "ljubljani": "Ljubljana", "ljubljana": "Ljubljana",
    "mariboru": "Maribor", "maribor": "Maribor",
    "celju": "Celje", "celje": "Celje",
    "kranju": "Kranj", "kranj": "Kranj"
  };

  const oneWord = raw.match(/^[\p{L}\p{M}\-]{3,}$/u);
  if (oneWord) {
    const key = t;
    return cityNormMap[key] || raw;
  }

  const m = raw.match(
    /\b(v|iz|pri|blizu|okolica)\s+([\p{L}\p{M}\-]{3,})(?:\s+([\p{L}\p{M}\-]{3,}))?(?:\s+([\p{L}\p{M}\-]{3,}))?/iu
  );
  if (m) {
    const parts = [m[2], m[3], m[4]].filter(Boolean);
    const joined = parts.join(" ");
    const joinedKey = joined.toLowerCase();
    return cityNormMap[joinedKey] || joined;
  }

  return null;
}

function missingSlot(p) {
  if (p.vehicleAge == null) return "vehicleAge";
  if (p.horsepower == null) return "horsepower";
  if (!p.coverageLevel) return "coverageLevel";
  return null;
}

function questionForSlot(slot) {
  if (slot === "vehicleAge") return "Koliko je star avto (v letih)?";
  if (slot === "horsepower") return "Koliko ima avto konjskih moči (KM)?";
  if (slot === "coverageLevel") return "Katero kritje želite: osnovno, delni kasko ali polni kasko?";
  return null;
}

function formatCoverageLabel(lvl) {
  if (lvl === "osnovno") return "osnovno zavarovanje";
  if (lvl === "delni_kasko") return "delni kasko";
  if (lvl === "polni_kasko") return "polni kasko";
  return lvl;
}

function looksLikeInfoQuestion(text) {
  const t = (text || "").toLowerCase().trim();
  const infoPatterns = [
    /^kaj\b/, /^kaj pomeni\b/, /^kaj je\b/, /^kako\b/, /^kdaj\b/, /^kje\b/, /^zakaj\b/,
    /pomeni/, /razlika/, /krije/, /kritje/, /postopek/, /prijavim škodo/, /odškodnin/, /odskodnin/, /dokaz/,
    /kaj potrebujem/, /kaj rabim/, /kateri dokument/, /škod/, /skod/
  ];
  return infoPatterns.some((p) => (p instanceof RegExp ? p.test(t) : t.includes(p)));
}

function hasCalcWording(text) {
  const t = (text || "").toLowerCase();
  return (
    t.includes("izračun") ||
    t.includes("izracun") ||
    t.includes("izračunaj") ||
    t.includes("izracunaj") ||
    t.includes("premija") ||
    t.includes("koliko stane") ||
    t.includes("koliko bi stal") ||
    t.includes("cena") ||
    t.includes("strošek") ||
    t.includes("stane") ||
    t.includes("koliko pride")
  );
}

function hasAnyPremiumSlotHints(text) {
  const t = (text || "").toLowerCase();
  const hasAge = /(\d{1,2})\s*(let|leto|leti|letih|lit)\b/i.test(t) || t.includes("star ");
  const hasHp = /(\d{2,4})\s*(km|konj|konji|konjev|konjska)\b/i.test(t) || t.includes("konj");
  const hasCity = /\b(v|iz|pri|blizu|okolica)\s+[\p{L}\p{M}\-]{3,}/iu.test(text || "");
  return hasAge || hasHp || hasCity;
}

function wantsComparison(text) {
  const t = (text || "").toLowerCase();
  return (
    t.includes("primerjaj") ||
    t.includes("primerjava") ||
    t.includes("primerjavo") ||
    t.includes("daj še") ||
    t.includes("še delni") ||
    t.includes("še polni") ||
    t.includes("še osnovno") ||
    t.includes("namesto") ||
    t.includes("tudi delni") ||
    t.includes("tudi polni")
  );
}

function isNewPremiumRequest(text) {
  const t = (text || "").toLowerCase();
  const hasAgeHint = /(\d{1,2})\s*(let|leto|leti|letih|lit)\b/i.test(t) || t.includes("star ");
  const hasHpHint = /(\d{2,4})\s*(km|konj|konji|konjev|konjska)\b/i.test(t) || t.includes("konj");
  const hasCityHint = /\b(v|iz|pri|blizu|okolica)\s+/i.test(t);
  const hasVehicle = t.includes("avto") || t.includes("vozilo");
  return hasVehicle && (hasAgeHint || hasHpHint || hasCityHint);
}

function wantsNewVehicle(text) {
  const t = (text || "").toLowerCase();
  return (
    t.includes("nov izračun") ||
    t.includes("nov izracun") ||
    t.includes("novo vozilo") ||
    t.includes("nov avto") ||
    t.includes("novo avto") ||
    t.includes("drugo vozilo") ||
    t.includes("drug avto")
  );
}

// ✅ ali je input verjetno “odgovor na primerjavo?”
function isLikelyAnswerToComparison(text) {
  return (
    isAffirmative(text) ||
    isNegative(text) ||
    !!extractCoverageLevel(text, { wantsPremium: true }) ||
    mentionsCoverage(text)
  );
}

function isPremiumIntent(text, state) {
  if (state?.pendingComparison) return true;

  const infoQ = looksLikeInfoQuestion(text);
  const calcW = hasCalcWording(text);
  const cov = mentionsCoverage(text);
  const slots = hasAnyPremiumSlotHints(text);
  const cmp = wantsComparison(text);

  if (infoQ && !calcW) return false;
  if (calcW) return true;
  if (cov && slots) return true;
  if (cmp && state?.lastPremium) return true;
  return false;
}

function toDocMeta(d) {
  const text = String(d?.text || "");
  const title = (d?.title || d?.id || "Dokument").toString();
  const preview = text.replace(/\s+/g, " ").trim().slice(0, 220);
  return {
    id: String(d?.id || ""),
    title,
    preview,
    bytes: Buffer.byteLength(text, "utf8")
  };
}

function excerpt(text, maxChars = 1400) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  return s.length > maxChars ? s.slice(0, maxChars) + " …" : s;
}

// -------- routes --------

app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

app.get("/rag/docs", (req, res) => {
  res.json({
    ok: true,
    count: docs.length,
    docs: docs.map(toDocMeta)
  });
});

app.get("/rag/docs/:id", (req, res) => {
  const id = String(req.params.id || "");
  const d = docs.find((x) => String(x.id) === id);
  if (!d) return res.status(404).json({ ok: false, error: "Doc not found" });

  res.json({
    ok: true,
    id: String(d.id),
    title: (d.title || d.id || "Dokument").toString(),
    text: String(d.text || "")
  });
});

app.post("/reset", (req, res) => {
  const sessionId = String(req.body.sessionId || "default");
  sessions.delete(sessionId);
  res.json({ ok: true });
});

app.post("/turn", upload.single("audio"), async (req, res) => {
  try {
    const sessionId = String(req.body.sessionId || "default");
    const sess = getSession(sessionId);
    const messages = sess.messages;
    const state = sess.state;

    if (!req.file?.buffer) return res.status(400).json({ error: "Missing audio file" });

    const transcript = await sttTranscribe(req.file.buffer, req.file.originalname || "audio.webm");
    messages.push({ role: "user", content: transcript });

    const newVehicleReq = wantsNewVehicle(transcript);
    if (newVehicleReq) {
      state.lastPremium = null;
      state.comparedLevels = new Set();
      state.pendingComparison = null;
      state.premium = null;
    }

    if (state.pendingComparison) {
      const opts = state.pendingComparison.options || [];

      const lower = (transcript || "").toLowerCase();
      const mentionsBoth = lower.includes("delni") && lower.includes("polni");
      if (opts.length === 2 && mentionsBoth) {
        const replyText = "Lahko izračunam oba. Kateri naj najprej: delni ali polni kasko?";
        messages.push({ role: "assistant", content: replyText });
        return await respondWithTTS(res, transcript, replyText);
      }

      if (looksLikeInfoQuestion(transcript) && !mentionsCoverage(transcript) && !hasCalcWording(transcript)) {
        state.pendingComparison = null;
      } else if (
        !isLikelyAnswerToComparison(transcript)
      ) {
        state.pendingComparison = null;
        const replyText =
          "Nisem prepričan, kaj želite. Ali želite primerjavo kritij (delni/polni kasko) " +
          "ali imate drugo vprašanje (npr. prijava škode)?";
        messages.push({ role: "assistant", content: replyText });
        return await respondWithTTS(res, transcript, replyText);
      } else {
        const saidLevel = extractCoverageLevel(transcript, { wantsPremium: true });
        if (saidLevel && opts.includes(saidLevel)) {
          state.pendingComparison = null;
          state.premium = {
            vehicleAge: state.lastPremium?.vehicleAge ?? null,
            horsepower: state.lastPremium?.horsepower ?? null,
            city: state.lastPremium?.city ?? null,
            coverageLevel: saidLevel,
            pendingSlot: null
          };
        } else if (isNegative(transcript)) {
          state.pendingComparison = null;
          const replyText = "V redu. Želite še kaj? Lahko naredim nov izračun za drugo vozilo.";
          messages.push({ role: "assistant", content: replyText });
          return await respondWithTTS(res, transcript, replyText);
        } else if (isAffirmative(transcript)) {
          if (opts.length === 1) {
            const chosen = opts[0];
            state.pendingComparison = null;
            state.premium = {
              vehicleAge: state.lastPremium?.vehicleAge ?? null,
              horsepower: state.lastPremium?.horsepower ?? null,
              city: state.lastPremium?.city ?? null,
              coverageLevel: chosen,
              pendingSlot: null
            };
          } else {
            const replyText = "Super — želite delni kasko ali polni kasko?";
            messages.push({ role: "assistant", content: replyText });
            return await respondWithTTS(res, transcript, replyText);
          }
        } else {
          const replyText = "Samo da preverim — želite delni kasko ali polni kasko?";
          messages.push({ role: "assistant", content: replyText });
          return await respondWithTTS(res, transcript, replyText);
        }
      }
    }

    const inPremiumFlow = state.premium !== null;
    const wantsPremium = isPremiumIntent(transcript, state);
    const newPremiumReq = wantsPremium && !inPremiumFlow && isNewPremiumRequest(transcript);

    if (newPremiumReq) {
      state.comparedLevels = new Set();
      state.lastPremium = null;
      state.pendingComparison = null;
    }

    // ---------------- PREMIUM FLOW ----------------
    if (inPremiumFlow || wantsPremium) {
      if (!state.premium) {
        const lp = state.lastPremium;
        const allowPrefill = !!lp && !newVehicleReq;

        state.premium = {
          vehicleAge: allowPrefill ? lp.vehicleAge : null,
          horsepower: allowPrefill ? lp.horsepower : null,
          city: allowPrefill ? lp.city : null,
          coverageLevel: null,
          pendingSlot: null
        };
      }

      const p = state.premium;

      let age = null, hp = null, city = null, lvl = null;

      if (!p.pendingSlot) {
        age = extractVehicleAge(transcript);
        hp = extractHorsepower(transcript);
        city = extractCity(transcript);
        lvl = extractCoverageLevel(transcript, { wantsPremium: true });
      } else {
        if (p.pendingSlot === "vehicleAge") age = extractVehicleAge(transcript);
        if (p.pendingSlot === "horsepower") hp = extractHorsepower(transcript);
        if (p.pendingSlot === "coverageLevel") lvl = extractCoverageLevel(transcript, { wantsPremium: true });
      }

      if (age != null) p.vehicleAge = age;
      if (hp != null) p.horsepower = hp;
      if (city) p.city = city;
      if (lvl) p.coverageLevel = lvl;

      const slot = missingSlot(p);
      if (slot) {
        p.pendingSlot = slot;
        const q = questionForSlot(slot);
        const replyText = "Seveda — za izračun premije potrebujem še eno informacijo. " + q;
        messages.push({ role: "assistant", content: replyText });
        return await respondWithTTS(res, transcript, replyText, { premium: { ...p, pending: true } });
      }

      if (!p.city) p.city = "Other";

      const result = calculate_premium(p.vehicleAge, p.horsepower, p.city, p.coverageLevel);

      state.comparedLevels.add(p.coverageLevel);

      const wantDelni = !state.comparedLevels.has("delni_kasko");
      const wantPolni = !state.comparedLevels.has("polni_kasko");

      let followUp = "";
      let pendingOptions = null;

      if (p.coverageLevel === "osnovno") {
        if (wantDelni || wantPolni) {
          followUp = "Želite primerjavo za delni ali polni kasko?";
          pendingOptions = [
            ...(wantDelni ? ["delni_kasko"] : []),
            ...(wantPolni ? ["polni_kasko"] : [])
          ];
        } else {
          followUp = "Želite še kaj? Lahko naredim nov izračun za drugo vozilo.";
        }
      } else if (p.coverageLevel === "delni_kasko") {
        if (wantPolni) {
          followUp = "Želite še izračun za polni kasko?";
          pendingOptions = ["polni_kasko"];
        } else {
          followUp = "Želite še kaj? Lahko naredim nov izračun za drugo vozilo.";
        }
      } else if (p.coverageLevel === "polni_kasko") {
        if (wantDelni) {
          followUp = "Želite še izračun za delni kasko?";
          pendingOptions = ["delni_kasko"];
        } else {
          followUp = "Želite še kaj? Lahko naredim nov izračun za drugo vozilo.";
        }
      } else {
        followUp = "Želite še kaj? Lahko naredim nov izračun za drugo vozilo.";
      }

      const replyText =
        `Ocena premije za ${formatCoverageLabel(p.coverageLevel)} ` +
        `(${p.vehicleAge} let star avto, ${p.horsepower} KM, ${p.city}) je približno ` +
        `${result.annual_eur} € na leto (okoli ${result.monthly_eur} € na mesec). ` +
        followUp;

      state.lastPremium = {
        vehicleAge: p.vehicleAge,
        horsepower: p.horsepower,
        city: p.city
      };

      state.premium = null;
      state.pendingComparison = pendingOptions ? { options: pendingOptions } : null;

      messages.push({ role: "assistant", content: replyText });
      return await respondWithTTS(res, transcript, replyText, { premiumResult: result });
    }

    // ---------------- RAG FLOW ----------------
    const hits = retrieveDocs(transcript, docs, 10); // ✅ 5–10 docs
    const contextDocs = hits.length ? hits : docs.slice(0, 10);

    const context = contextDocs
      .map((d) => `---\nDOC: ${d.id}\n${excerpt(d.text, 1400)}`)
      .join("\n");

    const ragSystemMsg = {
      role: "system",
      content:
        "PRAVILA (strogo):\n" +
        "1) Odgovarjaj na podlagi spodnjih dokumentov: uporabi jih kot vir resnice, ampak povzemaj samo relevantne informacije.\n" +
        "2) Ne dodajaj svojih korakov, ki niso iz dokumentov.\n" +
        "3) Če dokumenti ne vsebujejo odgovora, reci: 'Tega podatka v dokumentih nimam.' in vprašaj 1 dodatno vprašanje.\n" +
        "4) Če uporabnik sprašuje o postopku, vrni kratek odgovor (3–6 alinej max) in ponudi: 'Lahko razložim še podrobneje, želite?'\n" +
        "5) Če je izjava nejasna, uporabljaj standardne zavarovalniške izraze in NE ponavljaj nejasnega izraza uporabnika.\n" +
        "6) Pri nejasnih izrazih najprej razjasni pomen z enim vprašanjem.\n" +
        "7) Če vprašanje ni povezano z zavarovanjem, povej da si zavarovalniški agent in usmeri na: prijava škode, kritja (delni/polni kasko), odškodnina/dokazi, roki, samoudeležba, izračun premije.\n" +
        "8) Odgovarjaj customer-facing (v 2. osebi): 'lahko prijavite', 'potrebujete', 'priporočamo'. Ne uporabljaj 'agent naj...'.\n\n" +
        "DOKUMENTI:\n" +
        context
    };

    const replyText = await llmReply([...messages, ragSystemMsg]);

    messages.push({ role: "assistant", content: replyText });
    return await respondWithTTS(res, transcript, replyText, { ragDocs: contextDocs.map((d) => d.id) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
