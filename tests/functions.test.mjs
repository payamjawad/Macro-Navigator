import test from "node:test";
import assert from "node:assert/strict";
import {onRequestPost as chat, findSymbols, numberCheck} from "../site/functions/api/chat.js";
import {onRequestPost as refresh} from "../site/functions/api/refresh.js";
import {onRequestPost as watch} from "../site/functions/api/watchlist.js";

const summary = {data_to: "2026-11-30", strip: [{label: "VIX", value: "26.7", note: "Elevated"}],
  economy: {regime: "Late cycle", text: "x", pillars: [], recession_prob: 0.12, watch: []},
  market: {criteria: {level: "High risk", summary: "2 defensive signals are on together.",
    defensive: [{signal: "Trend broken", now: true, on_up: 0.6301, off_up: 0.7212, dip_on: -0.12, dip_off: -0.05, eps: 72, horizons: [3, 6, 12]}],
    short: [], recovery: [], watch: [], triggers: [{name: "Volatility capitulation", rule: "VIX closes a month at 30 or above", now: "VIX 26.7 on 24 Nov", gap: 3.3}]},
    call: {prob_up_12m: 0.6912, base_rate: 0.6912, basis: "historical base rate", confidence: "Low"}, analog: {n: 166, mean: 0.054, up: 0.67, episodes: []}, combo: null}};
const universe = [{s: "NVDA", n: "Nvidia", k: "stock", f: "NVDA"}, {s: "AAPL", n: "Apple Inc.", k: "stock", f: "AAPL"}, {s: "^NDX", n: "Nasdaq 100", k: "index", f: "_NDX"}];
const nvda = {symbol: "NVDA", name: "Nvidia", kind: "stock", sector: "Information Technology", headline: "Headwinds stacked",
  layers: [{name: "Own trend", text: "below its 200-day average", tone: "bad"}], horizons: [], metrics: {beta: 1.8, price: 140.5},
  fundamentals: {growth: 0.55, margin: 0.52}, risk_evidence: null, flips: [{name: "Back above", text: "Around 160.00"}], reading: "r"};
const files = {"/data/summary.json": summary, "/data/universe.json": universe, "/data/tickers/NVDA.json": nvda};
const env = (extra = {}) => ({ASSETS: {fetch: async u => { const p = new URL(u).pathname; return files[p] ? new Response(JSON.stringify(files[p])) : new Response("", {status: 404}); }}, ...extra});
const req = (body, access = true, path = "/api/chat") => new Request("https://mn.pages.dev" + path, {method: "POST",
  headers: {"Content-Type": "application/json", ...(access ? {"Cf-Access-Authenticated-User-Email": "me@example.com"} : {})}, body: JSON.stringify(body)});
const ask = q => ({messages: [{role: "user", text: q}]});

test("locked without Cloudflare Access", async () => {
  const r = await chat({request: req(ask("hi"), false), env: env({GEMINI_API_KEY: "k"})});
  assert.equal(r.status, 403);
  assert.match((await r.json()).text, /Cloudflare Access/);
});

test("no key: rule-based answer whose numbers all match the data", async () => {
  const d = await (await chat({request: req(ask("What's the stance?")), env: env()})).json();
  assert.match(d.text, /High risk/); assert.match(d.model, /Rule-based/);
  assert.ok(d.check.total >= 3); assert.deepEqual(d.check.unmatched, []);
});

test("Gemini answer: invented number is flagged, real ones pass, ticker looked up", async () => {
  let sent;
  globalThis.fetch = async (url, init) => { sent = JSON.parse(init.body);
    return new Response(JSON.stringify({candidates: [{content: {parts: [{text: "NVDA moves about 1.8× the market. Gains followed 63% of the time vs 72%. Odds are 69%. It could hit 412.5 soon."}]}}]})); };
  const d = await (await chat({request: req(ask("How does NVDA look?")), env: env({GEMINI_API_KEY: "k"})})).json();
  assert.match(d.model, /Gemini \(gemini-3.5-flash-lite\)/);
  assert.ok(d.looked.some(x => x.startsWith("NVDA")));
  assert.deepEqual(d.check.unmatched, ["412.5"]);
  assert.equal(d.check.total, 5);
  assert.match(sent.systemInstruction.parts[0].text, /"symbol":"NVDA"/);
  assert.ok(d.sources.some(s => s.href === "#stock/NVDA"));
});

test("retries without thinking control, moves to the next model on quota or retired models", async () => {
  let calls = [];
  globalThis.fetch = async (url, init) => { const b = JSON.parse(init.body); calls.push(url.split("/models/")[1].split(":")[0] + (b.generationConfig.thinkingConfig ? "+t" : ""));
    assert.equal(b.generationConfig.temperature, undefined);
    if (b.generationConfig.thinkingConfig) return new Response("thinking_level is not supported", {status: 400});
    return new Response(JSON.stringify({candidates: [{content: {parts: [{text: "thinking...", thought: true}, {text: "Stance is High risk."}]}}]})); };
  let d = await (await chat({request: req(ask("stance?")), env: env({GEMINI_API_KEY: "k"})})).json();
  assert.deepEqual(calls, ["gemini-3.5-flash-lite+t", "gemini-3.5-flash-lite"]); assert.equal(d.text, "Stance is High risk.");
  calls = [];
  globalThis.fetch = async (url, init) => { const m = url.split("/models/")[1].split(":")[0]; calls.push(m);
    if (m === "gemini-3.5-flash-lite") return new Response("quota", {status: 429});
    return new Response(JSON.stringify({candidates: [{content: {parts: [{text: "From the backup lite model."}]}}]})); };
  d = await (await chat({request: req(ask("stance?")), env: env({GEMINI_API_KEY: "k"})})).json();
  assert.deepEqual(calls, ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]);
  assert.ok(!calls.includes("gemini-3.1-flash-lite") && !calls.includes("gemini-3.5-flash"));
  assert.match(d.model, /gemini-2.5-flash-lite/);
  calls = [];
  d = await (await chat({request: req(ask("stance?")), env: env({GEMINI_API_KEY: "k", GEMINI_MODELS: "my-model"})})).json();
  assert.deepEqual(calls, ["my-model"]);
  globalThis.fetch = async () => new Response("quota", {status: 429});
  d = await (await chat({request: req(ask("stance?")), env: env({GEMINI_API_KEY: "k"})})).json();
  assert.match(d.model, /allowance for these models is used up/); assert.match(d.text, /High risk/);
});

test("ticker detection", () => {
  assert.deepEqual(findSymbols("How does NVDA look?", universe), ["NVDA"]);
  assert.deepEqual(findSymbols("what about nvidia and $aapl", universe), ["NVDA", "AAPL"]);
  assert.deepEqual(findSymbols("Is CPI high? What is the VIX doing?", universe), []);
  assert.deepEqual(findSymbols("nasdaq risk", universe), ["^NDX"]);
  assert.deepEqual(findSymbols("is it a buy?", universe, "NVDA"), ["NVDA"]);
});

test("number check ignores horizons, years and small counts", () => {
  const c = numberCheck("Over 12 months since 1980, 4 of 5 layers and the 200-day average; 63% vs 72%.", {a: 0.63, b: 0.72});
  assert.equal(c.total, 2); assert.deepEqual(c.unmatched, []);
});

test("refresh and watchlist call GitHub", async () => {
  let r = await refresh({request: req({}, true, "/api/refresh"), env: env()});
  assert.match((await r.json()).message, /REFRESH_TOKEN/);
  const seen = [];
  globalThis.fetch = async (url, init = {}) => { seen.push(`${init.method || "GET"} ${url}`);
    if (url.includes("/contents/") && !init.method) return new Response(JSON.stringify({sha: "abc", content: btoa("# list\nASML\n")}));
    if (url.includes("/contents/")) { assert.equal(atob(JSON.parse(init.body).content), "# list\nASML\nTSM\n"); return new Response("{}", {status: 200}); }
    return new Response(null, {status: 204}); };
  const e = env({GITHUB_TOKEN: "t", GITHUB_REPO: "me/macro-navigator"});
  r = await refresh({request: req({}, true, "/api/refresh"), env: e});
  assert.match((await r.json()).message, /Refresh started/);
  r = await watch({request: req({symbol: "tsm"}, true, "/api/watchlist"), env: e});
  assert.match((await r.json()).message, /TSM added/);
  r = await watch({request: req({symbol: "ASML"}, true, "/api/watchlist"), env: e});
  assert.match((await r.json()).message, /already/);
  assert.ok(seen.some(s => s.startsWith("POST") && s.includes("publish.yml/dispatches")));
});

test("index names are not treated as numbers", () => {
  const c = numberCheck("The S&P 500 and Nasdaq 100 fell; the Nasdaq-100 lagged by 63%.", {a: 0.63});
  assert.equal(c.total, 1); assert.deepEqual(c.unmatched, []);
});

import {onRequestGet as quote, metrics as qmetrics} from "../site/functions/api/quote.js";

function fakeChart(sym, n = 500, drift = 0.0004, beta = 1.5) {
  const t0 = 1700000000, ts = [], cl = [];
  let p = 100, m = 100;
  for (let i = 0; i < n; i++) {
    const mr = Math.sin(i * 0.7) * 0.01 + 0.0003;
    m *= 1 + mr; p *= 1 + beta * mr + drift * (sym === "^GSPC" ? 0 : -3);
    ts.push(t0 + i * 86400); cl.push(sym === "^GSPC" ? m : p);
  }
  return {chart: {result: [{meta: {symbol: sym, longName: sym === "^GSPC" ? "S&P 500" : "Zeta Corp"}, timestamp: ts,
    indicators: {quote: [{close: cl}], adjclose: [{adjclose: cl}]}}], error: null}};
}

test("live quote: builds a price-based view for an unlisted ticker", async () => {
  globalThis.fetch = async url => {
    const sym = decodeURIComponent(url.split("/chart/")[1].split("?")[0]);
    if (sym === "NOPE") return new Response(JSON.stringify({chart: {result: null, error: {code: "Not Found"}}}), {status: 404});
    return new Response(JSON.stringify(fakeChart(sym)));
  };
  const r = await quote({request: new Request("https://mn.pages.dev/api/quote?s=zeta"), env: env()});
  const v = await r.json();
  assert.equal(v.symbol, "ZETA"); assert.equal(v.name, "Zeta Corp"); assert.equal(v.live, true);
  assert.equal(v.layers.length, 4); assert.ok(v.metrics.beta > 1.2 && v.metrics.beta < 1.8, `beta ${v.metrics.beta}`);
  assert.equal(v.layers[1].tone, "bad");                     // market layer from summary: High risk
  const miss = await quote({request: new Request("https://mn.pages.dev/api/quote?s=NOPE"), env: env()});
  assert.equal(miss.status, 404);
  const bad = await quote({request: new Request("https://mn.pages.dev/api/quote?s=%3Cscript%3E"), env: env()});
  assert.equal(bad.status, 400);
});

test("chat uses a live-looked-up asset sent by the app", async () => {
  let sent;
  globalThis.fetch = async (url, init) => { sent = JSON.parse(init.body);
    return new Response(JSON.stringify({candidates: [{content: {parts: [{text: "ZETA moves about 1.5× the market."}]}}]})); };
  const asset = {symbol: "ZETA", name: "Zeta Corp", kind: "stock", layers: [], horizons: [], flips: [], metrics: {beta: 1.5}, reading: "r"};
  const d = await (await chat({request: req({messages: [{role: "user", text: "Explain the ZETA view"}], symbol: "ZETA", asset}),
    env: env({GEMINI_API_KEY: "k"})})).json();
  assert.ok(d.looked.includes("ZETA live prices"));
  assert.match(sent.systemInstruction.parts[0].text, /"symbol":"ZETA"/);
  assert.deepEqual(d.check.unmatched, []);
});

test("live quote falls back to Stooq when Yahoo refuses", async () => {
  const csv = sym => "Date,Open,High,Low,Close,Volume\n" + Array.from({length: 400}, (_, i) => {
    const d = new Date(Date.UTC(2025, 0, 1) + i * 86400000).toISOString().slice(0, 10);
    const base = sym === "^spx" ? 5000 : 20;
    return `${d},1,1,1,${(base * (1 + 0.001 * i + 0.01 * Math.sin(i))).toFixed(2)},100`;
  }).join("\n");
  globalThis.fetch = async url => {
    if (url.includes("yahoo")) return new Response("Too Many Requests", {status: 429});
    const s = decodeURIComponent(url.split("s=")[1].split("&")[0]);
    if (s === "nope.us") return new Response("No data", {status: 200});
    return new Response(csv(s));
  };
  const r = await quote({request: new Request("https://mn.pages.dev/api/quote?s=sofi"), env: env()});
  assert.equal(r.status, 200);
  const v = await r.json();
  assert.equal(v.symbol, "SOFI"); assert.ok(v.metrics.price > 0); assert.match(v.reading, /Stooq/);
  assert.equal(v.horizons[0].tag, "No proven edge");
  const miss = await quote({request: new Request("https://mn.pages.dev/api/quote?s=NOPE"), env: env()});
  assert.equal(miss.status, 404);
});
