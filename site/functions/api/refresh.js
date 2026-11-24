// POST /api/refresh - starts the GitHub "Publish" workflow; fresh data arrives in about 5-8 minutes.
const json = (obj, status = 200) => new Response(JSON.stringify(obj), {status, headers: {"Content-Type": "application/json", "Cache-Control": "no-store"}});
const accessOK = (request, env) => env.ALLOW_PUBLIC === "1" ||
  Boolean(request.headers.get("Cf-Access-Authenticated-User-Email") || request.headers.get("Cf-Access-Jwt-Assertion"));

export async function dispatch(env) {
  return fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/publish.yml/dispatches`, {
    method: "POST",
    headers: {Authorization: `Bearer ${env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "macro-navigator",
      "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"},
    body: JSON.stringify({ref: "main"})});
}

export async function onRequestPost({request, env}) {
  if (!accessOK(request, env)) return json({message: "Refresh is locked until the site sits behind Cloudflare Access (README, step 4)."}, 403);
  if (!env.GITHUB_TOKEN || !env.GITHUB_REPO) return json({message: "Refresh needs the REFRESH_TOKEN GitHub secret (README, step 3)."});
  const r = await dispatch(env);
  return json(r.status === 204
    ? {message: "Refresh started. New data in about 5–8 minutes; reopen the app then."}
    : {message: `GitHub refused the refresh (${r.status}). Check the REFRESH_TOKEN permissions.`});
}
