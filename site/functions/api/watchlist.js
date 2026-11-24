// POST /api/watchlist {symbol} - adds a ticker to config/watchlist.txt in the repository and starts a refresh.
import {dispatch} from "./refresh.js";
const json = (obj, status = 200) => new Response(JSON.stringify(obj), {status, headers: {"Content-Type": "application/json", "Cache-Control": "no-store"}});
const accessOK = (request, env) => env.ALLOW_PUBLIC === "1" ||
  Boolean(request.headers.get("Cf-Access-Authenticated-User-Email") || request.headers.get("Cf-Access-Jwt-Assertion"));

export async function onRequestPost({request, env}) {
  if (!accessOK(request, env)) return json({message: "The watchlist is locked until the site sits behind Cloudflare Access (README, step 4)."}, 403);
  if (!env.GITHUB_TOKEN || !env.GITHUB_REPO) return json({message: "The watchlist needs the REFRESH_TOKEN GitHub secret (README, step 3)."});
  const body = await request.json().catch(() => ({}));
  const sym = String(body.symbol || "").toUpperCase().trim();
  if (!/^[A-Z0-9.\-^=]{1,12}$/.test(sym)) return json({message: "That doesn't look like a ticker."}, 400);
  const api = `https://api.github.com/repos/${env.GITHUB_REPO}/contents/config/watchlist.txt`;
  const H = {Authorization: `Bearer ${env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "macro-navigator",
    "X-GitHub-Api-Version": "2022-11-28"};
  let sha, text = "";
  const cur = await fetch(api, {headers: H});
  if (cur.ok) { const d = await cur.json(); sha = d.sha; text = atob(String(d.content).replace(/\n/g, "")); }
  const have = text.split("\n").map(s => s.split("#")[0].trim().toUpperCase()).filter(Boolean);
  if (have.includes(sym)) return json({message: `${sym} is already on the watchlist; it appears after the next refresh.`});
  const next = (text.replace(/\s*$/, "") + "\n" + sym + "\n").replace(/^\n/, "");
  const put = await fetch(api, {method: "PUT", headers: {...H, "Content-Type": "application/json"},
    body: JSON.stringify({message: `Add ${sym} to watchlist`, content: btoa(next), ...(sha ? {sha} : {})})});
  if (!put.ok) return json({message: `GitHub refused the change (${put.status}). Check the REFRESH_TOKEN permissions.`});
  await dispatch(env);
  return json({message: `${sym} added. A refresh has started; it appears in about 5–8 minutes.`});
}
