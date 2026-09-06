/**
 * R.119 Step 0.5 — Node + axios 1.18.1 (worker's HTTP client)
 * CommonJS variant — needs `require()` to find axios via NODE_PATH.
 *
 * Run from any cwd:
 *   NODE_PATH=/Volumes/SSDNSKIY/VSCODE/reelant/node_modules/.pnpm/axios@1.18.1/node_modules \
 *     node r119_web_timeout_test.cjs 200
 */
const axios = require("axios");

const URL = "https://anton722451--comfyui-minimax-h3-r119-test-serve.modal.run";
const PER_REQUEST_TIMEOUT_MS = 900_000;

async function runTest(sleepSeconds) {
  const t0 = Date.now();
  let status, body, errorMsg, redirectCount, finalHeaders, redirectUrls;

  try {
    const res = await axios.post(
      `${URL}/api/test-run`,
      { sleep: sleepSeconds },
      {
        headers: { "Content-Type": "application/json" },
        maxRedirects: 21,
        timeout: PER_REQUEST_TIMEOUT_MS,
        validateStatus: () => true,
      }
    );
    status = res.status;
    body = res.data;
    finalHeaders = res.headers;
    const redirectable = res.request;
    redirectCount = redirectable?._redirectCount ?? 0;
    redirectUrls = redirectable?._redirects?.map((r) => r.url) ?? [];
  } catch (err) {
    status = err.response?.status ?? "NETWORK_ERROR";
    body = err.response?.data ?? null;
    finalHeaders = err.response?.headers;
    redirectCount = err.response?.request?._redirectCount ?? 0;
    redirectUrls = err.response?.request?._redirects?.map((r) => r.url) ?? [];
    errorMsg = `${err.code ?? "?"}: ${err.message}`;
  }

  const elapsedMs = Date.now() - t0;
  console.log(
    JSON.stringify(
      {
        sleepRequested: sleepSeconds,
        elapsedMs,
        elapsedSeconds: (elapsedMs / 1000).toFixed(1),
        status,
        redirectCount,
        redirectUrls,
        contentType: finalHeaders?.["content-type"],
        body,
        errorMsg,
      },
      null,
      2
    )
  );
}

const sleepArg = Number(process.argv[2] ?? 200);
runTest(sleepArg).catch((e) => {
  console.error("FATAL", e);
  process.exit(1);
});
