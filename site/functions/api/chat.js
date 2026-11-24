// POST /api/chat - Ask Navigator. Gemini writes the words; every number must come from the published data.
// Lite models only, and never gemini-3.1-flash-lite (kept free for the owner's other project). Google counts free
// limits per project and per model, so these do not touch that allowance. Override with a GEMINI_MODELS variable.
const DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"];
const STOP = new Set(["A", "I", "AN", "AND", "OR", "THE", "US", "USA", "UK", "EU", "AI", "CPI", "PCE", "GDP", "FED", "VIX", "ETF", "IPO",
  "CEO", "SP", "S", "P", "SPX", "YOY", "EPS", "PE", "OK", "IS", "IT", "IN", "ON", "AT", "TO", "OF", "BE", "DO", "MY", "ME", "WE", "NO",
  "SO", "IF", "AM", "PM", "Q1", "Q2", "Q3", "Q4", "HY", "BP", "NOW", "YES", "BUY", "SELL", "HOLD", "WHAT", "HOW", "WHY", "RISK"]);

export function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {status, headers: {"Content-Type": "application/json", "Cache-Control": "no-store"}});
}
export function accessOK(request, env) {
  if (env.ALLOW_PUBLIC === "1") return true;
  return Boolean(request.headers.get("Cf-Access-Authenticated-User-Email") || request.headers.get("Cf-Access-Jwt-Assertion"));
}
const share = x => (x == null || !isFinite(x) ? "n/a" : Math.round(x * 100) + "%");

export function findSymbols(text, universe, current) {
  const bySym = new Map(universe.map(u => [u.s.toUpperCase(), u]));
  const byName = new Map();
  for (const u of universe) {
    if (u.k !== "stock" || !u.n) continue;
    const w = u.n.toLowerCase().split(/[^a-z0-9]+/)[0];
    if (w && w.length >= 4) byName.set(w, byName.has(w) ? null : u.s);   // null = ambiguous
  }
  const out = [];
  const add = s => { if (s && bySym.has(s) && !out.includes(s)) out.push(s); };
  if (/nasdaq/i.test(text)) add("^NDX");
  for (const m of text.matchAll(/\$?[A-Za-z][A-Za-z.\-]{0,6}/g)) {
    const raw = m[0];
    if (raw.startsWith("$")) { add(raw.slice(1).toUpperCase()); continue; }
    if (raw === raw.toUpperCase() && raw.length >= 2 && !STOP.has(raw)) add(raw.replace(".", "-"));
    const n = byName.get(raw.toLowerCase());
    if (n) add(n);
  }
  if (!out.length && current && /\b(it|this|that|the stock|its)\b/i.test(text)) add(String(current).toUpperCase());
  return out.slice(0, 2);
}

export function compact(summary, assets) {
  const m = summary.market || {}, c = m.criteria || {}, call = m.call || {}, a = m.analog || {}, e = summary.economy || {};
  const pick = d => ({signal: d.signal, on_now: d.now, rose_within_12m_when_on: d.on_up, rose_within_12m_otherwise: d.off_up,
    avg_worst_dip_when_on: d.dip_on, avg_worst_dip_otherwise: d.dip_off, episodes: d.eps, proven_at_months: d.horizons});
  return {
    data_to: summary.data_to,
    market_strip: (summary.strip || []).map(x => `${x.label} ${x.value} (${x.note})`),
    economy: {regime: e.regime, summary: e.text, pillars: (e.pillars || []).map(p => `${p.name}: ${p.label} - ${p.detail}`),
      recession_probability_12m: e.recession_prob, would_change_the_read: e.watch},
    sp500: {stance: c.level, stance_summary: c.summary, defensive: (c.defensive || []).map(pick), short_term_warnings: (c.short || []).map(pick),
      recovery_reentry_signals: (c.recovery || []).map(pick), watch_only_unproven: c.watch, reentry_triggers: c.triggers,
      odds_higher_in_12m: {probability: call.prob_up_12m, base_rate: call.base_rate, basis: call.basis, confidence: call.confidence},
      similar_past_months: {count: a.n, avg_next_12m: a.mean, share_positive: a.up, p10: a.p10, p90: a.p90,
        closest: (a.episodes || []).map(x => ({month: x.closest, next_12m: x.next12}))},
      combination_now: m.combo ? {signals: m.combo.signals, episodes: m.combo.eps, avg_next_12m_when_on: m.combo.on_mean,
        avg_next_12m_otherwise: m.combo.off_mean, history: m.combo.episodes_list} : null},
    assets: assets.map(t => ({symbol: t.symbol, name: t.name, kind: t.kind, sector: t.sector, headline: t.headline,
      layers: t.layers.map(l => `${l.name}: ${l.text} [${({good: "tailwind", bad: "headwind", warn: "mixed"})[l.tone] || "n/a"}]`),
      horizons: t.horizons, metrics: t.metrics, fundamentals: t.fundamentals, risk_evidence_in_months_like_now: t.risk_evidence,
      what_would_flip_it: t.flips.map(f => `${f.name}: ${f.text}`), reading: t.reading})),
  };
}

function numbersIn(obj, out = []) {
  if (typeof obj === "number" && isFinite(obj)) out.push(obj);
  else if (typeof obj === "string") for (const m of obj.matchAll(/[-−]?\d[\d,]*(?:\.\d+)?/g)) out.push(parseFloat(m[0].replace(/,/g, "").replace("−", "-")));
  else if (obj && typeof obj === "object") Object.values(obj).forEach(v => numbersIn(v, out));
  return out;
}
export function numberCheck(text, context) {
  const known = numbersIn(context).flatMap(k => [Math.abs(k), Math.abs(k * 100)]);
  const found = [];
  const re = /([-−+]?\$?\d[\d,]*(?:\.\d+)?)(\s*%|\s*bp|\s*×|x\b)?(\s*[-–]?\s*(?:day|month|year|week|quarter)s?\b)?/gi;
  const clean = String(text).replace(/S&P\s*500|Nasdaq[\s-]*100|Russell\s*2000|Dow\s*30|FTSE\s*100|Stoxx\s*600/gi, " ");
  for (const m of clean.matchAll(re)) {
    if (m[3]) continue;                                       // horizons such as "12-month" or "200-day"
    const v = parseFloat(m[1].replace(/[$,+]/g, "").replace("−", "-"));
    if (!isFinite(v)) continue;
    const isPct = Boolean(m[2] && m[2].includes("%"));
    if (!isPct && Number.isInteger(v) && Math.abs(v) <= 12) continue;    // small counts ("4 of 5 layers")
    if (!isPct && Number.isInteger(v) && v >= 1900 && v <= 2100) continue; // years
    found.push({s: m[0].trim(), v: Math.abs(v)});
  }
  const unmatched = found.filter(f => !known.some(k => Math.abs(k - f.v) <= Math.max(0.051, f.v * 0.006)));
  return {total: found.length, matched: found.length - unmatched.length, unmatched: unmatched.map(u => u.s).slice(0, 6)};
}

export function fallback(summary, assets) {
  const c = summary.market?.criteria || {}, call = summary.market?.call || {};
  let t = `Market stance: ${c.level}. ${c.summary || ""}`;
  const def = (c.defensive || []).filter(d => d.now).map(d => `${d.signal} (gains followed ${share(d.on_up)} of the time vs ${share(d.off_up)})`);
  if (def.length) t += `\n\nDefensive signals on: ${def.join("; ")}.`;
  t += `\n\nOdds the S&P 500 is higher in 12 months: ${share(call.prob_up_12m)}, from the ${call.basis || "base rate"}.`;
  if ((c.triggers || []).length) t += `\n\nRe-entry triggers: ${c.triggers.map(x => `${x.rule} (now ${x.now})`).join("; ")}.`;
  for (const a of assets) t += `\n\n${a.symbol}: ${a.headline}. ${a.reading}`;
  return t;
}

function systemPrompt(ctx) {
  return `You are Ask Navigator, the research assistant inside the user's Macro Navigator app. Answer ONLY from DATA (JSON, data to ${ctx.data_to}).
Rules:
- Use only numbers that appear in DATA. Fractions are shares (0.63 = 63%). Never invent prices, dates or statistics; if DATA lacks something, say so.
- Keep horizons apart: 1-3 months, 6-12 months, 3-5 years. Say whether a point is proven by the app's evidence tests, research-backed, or has no proven edge.
- The app's 12-month models have no proven edge unless DATA says otherwise. Do not predict prices or tell the user to buy or sell; explain risks, odds and what would change the view.
- Plain language, short paragraphs, at most 170 words, no headings, tables or disclaimers (the app shows one).
DATA:
${JSON.stringify(ctx)}`;
}

async function gemini(key, system, messages, models = DEFAULT_MODELS) {
  const contents = messages.filter(m => m && m.text).slice(-10)
    .map(m => ({role: m.role === "assistant" ? "model" : "user", parts: [{text: String(m.text).slice(0, 2000)}]}));
  let lastErr = "", quotaHits = 0;
  for (const model of models) {
    // newer models reject or ignore temperature, so none is sent; light thinking keeps answers quick
    for (const config of [{maxOutputTokens: 2048, thinkingConfig: {thinkingLevel: "low"}}, {maxOutputTokens: 2048}]) {
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
        method: "POST", headers: {"Content-Type": "application/json", "x-goog-api-key": key},
        body: JSON.stringify({systemInstruction: {parts: [{text: system}]}, contents, generationConfig: config})});
      if (r.ok) {
        const d = await r.json();
        const text = (d.candidates?.[0]?.content?.parts || []).filter(p => !p.thought).map(p => p.text || "").join("").trim();
        if (text) return {text, model};
        lastErr = "empty answer";
        break;
      }
      lastErr = String(r.status);
      if (r.status === 400 && config.thinkingConfig) continue;   // model without this setting: retry plainly
      if (r.status === 429) quotaHits++;
      break;                                                      // 404, 429 or other: try the next model
    }
  }
  if (quotaHits) throw new Error("Gemini's free allowance for these models is used up for now; daily limits reset at midnight Pacific time");
  throw new Error(`Gemini unavailable (${lastErr})`);
}

export async function onRequestPost({request, env}) {
  if (!accessOK(request, env)) {
    return json({text: "Chat is locked until this site sits behind Cloudflare Access (README, step 4), so nobody else can use your Gemini key.", model: "Locked"}, 403);
  }
  let body;
  try { body = await request.json(); } catch (e) { return json({error: "Bad request"}, 400); }
  const messages = Array.isArray(body.messages) ? body.messages.slice(-10) : [];
  const question = [...messages].reverse().find(m => m.role === "user")?.text || "";
  const asset = async path => {
    const r = await env.ASSETS.fetch(new URL(path, request.url));
    return r.ok ? r.json() : null;
  };
  const summary = await asset("/data/summary.json");
  if (!summary) return json({text: "The data files are missing; run the Publish workflow once."});
  const universe = (await asset("/data/universe.json")) || [];
  const files = new Map(universe.map(u => [u.s.toUpperCase(), u.f]));
  const looked = ["S&P 500 stance", "re-entry triggers"];
  if (/econom|inflation|fed|rate|job|recession|growth/i.test(question)) looked.unshift("economy");
  const assets = [];
  for (const s of findSymbols(question, universe, body.symbol)) {
    const t = await asset(`/data/tickers/${files.get(s)}.json`);
    if (t) { assets.push(t); looked.push(`${s} prices, sector${t.kind === "stock" ? ", SEC filings" : ""}`); }
  }
  const sources = [{label: "S&P 500 stance", href: "#sp500"}];
  if (looked.includes("economy")) sources.push({label: "Economy", href: "#economy"});
  for (const t of assets) sources.push({label: `${t.symbol} layers`, href: t.symbol === "^NDX" ? "#nasdaq" : `#stock/${encodeURIComponent(t.symbol)}`});
  const ctx = compact(summary, assets);
  if (!env.GEMINI_API_KEY) {
    const text = fallback(summary, assets);
    return json({text, looked, sources, check: numberCheck(text, ctx), model: "Rule-based answer (add the GEMINI_API_KEY secret for AI answers)"});
  }
  try {
    const models = env.GEMINI_MODELS ? String(env.GEMINI_MODELS).split(",").map(m => m.trim()).filter(Boolean) : DEFAULT_MODELS;
    const {text, model} = await gemini(env.GEMINI_API_KEY, systemPrompt(ctx), messages, models);
    return json({text, looked, sources, check: numberCheck(text, ctx), model: `Gemini (${model})`});
  } catch (e) {
    const text = fallback(summary, assets);
    return json({text, looked, sources, check: numberCheck(text, ctx), model: `Rule-based answer: ${e.message}`});
  }
}
