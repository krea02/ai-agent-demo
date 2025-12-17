import fs from "fs";
import path from "path";

function tokenize(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

export function loadDocs(docsDir) {
  const files = fs.readdirSync(docsDir).filter((f) => f.endsWith(".md"));
  return files.map((f) => {
    const full = path.join(docsDir, f);
    const text = fs.readFileSync(full, "utf8");
    const tokenSet = new Set(tokenize(text));
    return { id: f, text, tokenSet };
  });
}

export function retrieveDocs(query, docs, k = 10) {
  const q = new Set(tokenize(query));

  const scored = docs.map((d) => {
    let score = 0;
    for (const w of q) if (d.tokenSet?.has(w)) score += 1;
    return { ...d, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .filter((x) => x.score > 0);
}
