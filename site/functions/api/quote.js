// GET /api/quote?s=SYMBOL - live, price-based view for a ticker that is not in the daily list.
// Same shape as the pre-built ticker files, without the sector and SEC layers (those arrive once it is on the watchlist).
const json = (obj, status = 200) => new Response(JSON.stringify(obj), {status, headers: {"Content-Type": "application/json", "Cache-Control": "max-age=900"}});
const UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"};

// Stooq: free daily prices without a key, used when Yahoo refuses requests from cloud servers.
const STOOQ = {"^GSPC": "^spx", "^NDX": "^ndx", "^IXIC": "^ndq", "^DJI": "^dji"};
export async function stooq(sym, fetcher = fetch) {
  const s = STOOQ[sym] || (sym.toLowerCase() + (sym.includes(".") || sym.startsWith("^") ? "" : ".us"));
  let r;
  try { r = await fetcher(`https://stooq.com/q/d/l/?s=${encodeURIComponent(s)}&i=d`, {headers: UA}); } catch (e) { return null; }
  if (!r || !r.ok) return null;
  const lines = (await r.text()).trim().split(/\r?\n/);
  if (lines.length < 61 || !/^date,/i.test(lines[0])) return null;
  const head = lines[0].toLowerCase().split(","), iD = head.indexOf("date"), iC = head.indexOf("close");
  const pts = lines.slice(1).slice(-520).map(l => l.split(",")).map(c => [Date.parse(c[iD] + "T21:00:00Z") / 1000, parseFloat(c[iC])])
    .filter(p => isFinite(p[0]) && isFinite(p[1]));
  return pts.length >= 60 ? {meta: {symbol: sym, shortName: sym, source: "stooq"}, pts} : null;
}

export async function chart(sym, fetcher = fetch) {
  for (const host of ["query1", "query2"]) {
    let r;
    try { r = await fetcher(`https://${host}.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?range=2y&interval=1d&includeAdjustedClose=true`, {headers: UA}); }
    catch (e) { continue; }
    if (!r.ok) continue;
    let d;
    try { d = await r.json(); } catch (e) { continue; }
    const res = d?.chart?.result?.[0];
    if (!res || !res.timestamp) continue;
    const close = res.indicators?.adjclose?.[0]?.adjclose || res.indicators?.quote?.[0]?.close || [];
    const pts = res.timestamp.map((t, i) => [t, close[i]]).filter(p => p[1] != null && isFinite(p[1]));
    if (pts.length) return {meta: res.meta || {}, pts};
  }
  return stooq(sym, fetcher);
}

const ret = (p, n) => (p.length > n ? p[p.length - 1] / p[p.length - 1 - n] - 1 : null);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

export function metrics(pts, mkt) {
  const p = pts.map(x => x[1]);
  const hi = Math.max(...p.slice(-252));
  const sma200 = p.length >= 200 ? mean(p.slice(-200)) : null;
  const m = {price: +p[p.length - 1].toFixed(2), date: new Date(pts[pts.length - 1][0] * 1000).toISOString().slice(0, 10),
    chg_1d: ret(p, 1), ret_1m: ret(p, 21), ret_3m: ret(p, 63), ret_12m: ret(p, 252), from_high: p[p.length - 1] / hi - 1,
    sma200: sma200 && +sma200.toFixed(2), above_200d: sma200 == null ? null : p[p.length - 1] > sma200};
  if (mkt) {
    const week = arr => { const out = new Map(); for (const [t, v] of arr) out.set(Math.floor((t / 86400 + 3) / 7), v); return out; };  // last close each week
    const a = week(pts), b = week(mkt.pts), keys = [...a.keys()].filter(k => b.has(k)).sort((x, y) => x - y).slice(-105);
    const ra = [], rb = [];
    for (let i = 1; i < keys.length; i++) { ra.push(a.get(keys[i]) / a.get(keys[i - 1]) - 1); rb.push(b.get(keys[i]) / b.get(keys[i - 1]) - 1); }
    if (ra.length >= 30) {
      const ma = mean(ra), mb = mean(rb);
      let cov = 0, va = 0, vb = 0;
      for (let i = 0; i < ra.length; i++) { cov += (ra[i] - ma) * (rb[i] - mb); va += (ra[i] - ma) ** 2; vb += (rb[i] - mb) ** 2; }
      m.beta = +(cov / vb).toFixed(2);
      m.r2 = +((cov * cov) / (va * vb)).toFixed(2);
    }
    const s3 = ret(mkt.pts.map(x => x[1]), 63);
    m.rel_3m = m.ret_3m != null && s3 != null ? m.ret_3m - s3 : null;
  }
  return m;
}

export function view(sym, meta, m, summary) {
  const econ = summary.economy || {}, lvl = summary.market?.criteria?.level || "";
  const trend = m.above_200d == null || m.ret_3m == null ? "neutral" : (m.above_200d && m.ret_3m > 0 ? "good" : (!m.above_200d && m.ret_3m < 0 ? "bad" : "warn"));
  const pc = x => (x > 0 ? "+" : "") + Math.round(x * 100) + "%";
  const layers = [
    {name: "Economy", text: econ.regime || "n/a", tone: econ.tone || "neutral"},
    {name: "Market (S&P 500)", text: lvl + (lvl === "High risk" || lvl === "Elevated risk" ? ", no re-entry signal yet" : ""),
      tone: {"High risk": "bad", "Elevated risk": "warn", "Normal": "good", "Recovery window": "good"}[lvl] || "neutral"},
    {name: "Own trend", tone: trend, text: `${pc(m.from_high)} from its 52-week high; ${m.above_200d ? "above" : "below"} its 200-day average`
      + (m.ret_3m != null ? `; 3-month return ${pc(m.ret_3m)}` : "")},
    {name: "Sensitivity", tone: (m.beta ?? 1) > 1.1 ? "warn" : "neutral", text: m.beta != null ? `Moves about ${m.beta.toFixed(1)}× the S&P 500` : "Not enough history"},
  ];
  const nb = layers.filter(l => l.tone === "bad").length, ng = layers.filter(l => l.tone === "good").length;
  const head = nb >= 3 ? ["Headwinds stacked", "bad"] : ng >= 3 ? ["Tailwinds aligned", "good"] : ["Mixed signals", "warn"];
  const short = trend === "good" ? ["Momentum with it", "good"] : trend === "bad" ? ["Momentum against it", "bad"] : ["No clear momentum", "neutral"];
  const flips = [];
  if (m.sma200) flips.push(m.above_200d ? {name: "A close below its 200-day average", text: `Around ${m.sma200.toLocaleString()}; turns the trend layer negative.`}
    : {name: "Back above its 200-day average", text: `Around ${m.sma200.toLocaleString()}; turns the trend layer positive.`});
  flips.push({name: "A market re-entry trigger fires", text: "VIX month-end ≥ 30 or credit spread +0.5 pts in 6 months."});
  flips.push({name: "Add it to the watchlist", text: "The next refresh adds its sector and SEC fundamentals layers."});
  return {symbol: sym, name: meta.longName || meta.shortName || sym, kind: "stock", sector: null, live: true, metrics: m, layers,
    headline: head[0], headline_tone: head[1],
    horizons: [{h: "1–3 months", text: short[0].replace("Momentum", "Own trend") + " (not a tested edge)", tone: "neutral", tag: "No proven edge"},
      {h: "6–12 months", text: "Unclear: layers disagree", tone: "neutral", tag: "No proven edge"},
      {h: "3–5 years", text: "Set by the business", tone: "warn", tag: "Not modelled yet"}],
    flips, risk_evidence: null,
    reading: `Live lookup from prices${meta.source === "stooq" ? " (Stooq)" : ""}. ${nb} of ${layers.length} layers lean against it and ${ng} support it`
      + (m.beta != null ? `; it tends to move about ${m.beta.toFixed(1)}× the market` : "") + ". In 2026 tests none of these layers predicted single-stock returns: use them for context and size, not timing."};
}

export async function onRequestGet({request, env}) {
  const sym = (new URL(request.url).searchParams.get("s") || "").toUpperCase().trim();
  if (!/^[A-Z0-9.\-^=]{1,12}$/.test(sym)) return json({error: "Not a ticker"}, 400);
  const [c, mkt, sres] = await Promise.all([chart(sym), chart("^GSPC"), env.ASSETS.fetch(new URL("/data/summary.json", request.url))]);
  if (!c || c.pts.length < 60) return json({error: `No price history found for ${sym}`}, 404);
  const summary = sres.ok ? JSON.parse((await sres.text()).replace(/\bNaN\b/g, "null")) : {};
  return json(view(sym, c.meta, metrics(c.pts, mkt), summary));
}
