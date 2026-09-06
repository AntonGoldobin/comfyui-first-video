/**
 * R.119 Step 0.5 — verify Modal Web Function /api/test-run 150s redirect behavior
 * using the REAL worker HTTP client (Node + axios 1.18.1, same version as worker).
 *
 * Mirrors `reelant/apps/worker/src/providers/modal-comfyui.provider.ts`:
 *   - uses @nestjs/axios HttpService (under the hood: axios 1.18.1)
 *   - axios Node defaults: maxRedirects=21, no global timeout
 *   - per-request timeout = SUBMIT_AXIOS_TIMEOUT_MS (900_000 in current worker)
 *
 * Nest's HttpService does NOT override axios defaults. We import axios directly
 * — identical HTTP client, identical config surface.
 *
 * Tests:
 *   1. sleep=200 — under nominal R.148 worst case
 *   2. sleep=400 — exceeds typical R.148 worst case (~548s)
 *   3. sleep=550 — R.148 observed worst case
 *
 * Per-test output: elapsedMs, HTTP status, _redirectCount, body, error.
 */

import axios from "axios";

const URL = "https://anton722451--comfyui-minimax-h3-r119-test-serve.modal.run";
const PER_REQUEST_TIMEOUT_MS = 900_000;

async function runTest(sleepSeconds) {
  const t0 = Date.now();
  let status, body, errorMsg, redirectCount, finalHeaders;

  try {
    const res = await axios.post(
      `${URL}/api/test-run`,
      { sleep: sleepSeconds },
      {
        headers: { "Content-Type": "application/json" },
        maxRedirects: 21,        // axios default — explicit for clarity
        timeout: PER_REQUEST_TIMEOUT_MS,
        // Don't validate status — we want to see the raw response code on redirects too
        validateStatus: () => true,
      }
    );
    status = res.status;
    body = res.data;
    finalHeaders = res.headers;
    // axios 1.18 + follow-redirects 1.16 — request object IS the Redirectable
    const redirectable = res.request;
    redirectCount = redirectable?._redirectCount ?? 0;
  } catch (err) {
    status = err.response?.status ?? "NETWORK_ERROR";
    body = err.response?.data ?? null;
    finalHeaders = err.response?.headers;
    redirectCount = err.response?.request?._redirectCount ?? 0;
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
