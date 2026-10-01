#!/usr/bin/env bash
#
# Keke route smoke test — exercises every HTTP route against a running server.
#
#   ./scripts/smoke_test.sh [port]
#
# Exits non-zero if any route fails. Start the server first:
#   npm start
#
# Routes that require third-party credentials (AWS, Google, Dropbox) are expected
# to return a "not configured" payload rather than a 500; the script accepts that.

set -uo pipefail

PORT="${1:-3000}"
BASE="http://localhost:${PORT}"
PASS=0
FAIL=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

source .venv/bin/activate 2>/dev/null || true

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }

# check <name> <expected-status> <curl args...>
# expected-status may be an alternation, e.g. "200|400".
# Extra env: ALLOW_BODY — a regex that must match the body (optional).
check() {
    local name="$1" expect="$2"; shift 2
    local body status
    body="$(curl -s -m 120 -w $'\n__STATUS__%{http_code}' "$@" 2>&1)"
    status="${body##*__STATUS__}"
    body="${body%$'\n'__STATUS__*}"

    if [[ "$status" =~ ^($expect)$ ]]; then
        if [[ -n "${ALLOW_BODY:-}" ]] && ! grep -qE "$ALLOW_BODY" <<<"$body"; then
            red "FAIL  $name (status $status, body did not match /$ALLOW_BODY/)"
            printf '      %s\n' "$(head -c 300 <<<"$body")"
            FAIL=$((FAIL + 1))
        else
            green "PASS  $name ($status)"
            PASS=$((PASS + 1))
        fi
    else
        red "FAIL  $name (expected $expect, got $status)"
        printf '      %s\n' "$(head -c 300 <<<"$body")"
        FAIL=$((FAIL + 1))
    fi
    unset ALLOW_BODY
}

# --- Build a fixture workbook -----------------------------------------------
python - "$TMP" <<'PY' 2>/dev/null
import sys, pandas as pd
tmp = sys.argv[1]
df = pd.DataFrame({
    "price": [10.0, 12.5, 11.0, 30.0, 13.0, 12.0, 11.5, 29.0, 14.0, 13.5],
    "volume": [100, 200, 150, 900, 180, 170, 160, 880, 190, 185],
    "label":  ["a", "b", "a", "c", "b", "a", "b", "c", "a", "b"],
})
df.to_excel(f"{tmp}/fixture.xlsx", index=False, sheet_name="Sheet1")
df.to_csv(f"{tmp}/fixture.csv", index=False)
PY

if [[ ! -f "$TMP/fixture.xlsx" ]]; then
    red "FATAL: could not build fixture workbook"
    exit 1
fi

echo "=== Keke smoke test against $BASE ==="
echo

# --- Health / static --------------------------------------------------------
check "GET  /health"                    200 "$BASE/health"
check "GET  / (index.html)"             200 "$BASE/"
check "GET  /api/nonexistent -> 404"    404 "$BASE/api/nonexistent"

# --- Upload -----------------------------------------------------------------
UP="$(curl -s -m 120 -F "file=@$TMP/fixture.xlsx" "$BASE/api/excel/upload")"
SID="$(python -c "import sys,json;print(json.load(sys.stdin).get('session_id',''))" <<<"$UP" 2>/dev/null)"
if [[ -z "$SID" ]]; then
    red "FATAL: upload did not return a session_id"
    echo "$UP" | head -c 400
    exit 1
fi
green "PASS  POST /api/excel/upload (session $SID)"
PASS=$((PASS + 1))
SHEET="Sheet1"
J="Content-Type: application/json"

# --- Core spreadsheet routes ------------------------------------------------
check "GET  analyze"          200 "$BASE/api/excel/$SID/analyze/$SHEET"
check "GET  summary"          200 "$BASE/api/excel/$SID/summary"
check "GET  preview"          200 "$BASE/api/excel/$SID/preview/$SHEET?limit=5"
check "GET  columns"          200 "$BASE/api/excel/$SID/columns/$SHEET"
check "POST validate"         200 -X POST -H "$J" -d '{"rules":[]}' \
                             "$BASE/api/excel/$SID/validate/$SHEET"
check "POST transform"        200 -X POST -H "$J" -d '{"transformations":[]}' \
                             "$BASE/api/excel/$SID/transform/$SHEET"
check "POST clean"            200 -X POST -H "$J" -d '{"operations":[{"type":"remove_duplicates"}]}' \
                             "$BASE/api/excel/$SID/clean/$SHEET"
check "POST chart-preview"    200 -X POST -H "$J" \
                             -d '{"chart_config":{"type":"bar","title":"t","x":"price","y":"volume"}}' \
                             "$BASE/api/excel/$SID/chart-preview/$SHEET"
check "POST formulas"         200 -X POST -H "$J" -d '{"formulas":{"total":"price * volume"}}' \
                             "$BASE/api/excel/$SID/formulas/$SHEET"
for f in csv json excel parquet; do
    check "GET  export ($f)" 200 "$BASE/api/excel/$SID/export/$SHEET?format=$f"
done

# --- ML ---------------------------------------------------------------------
check "POST predict"      200 -X POST -H "$J" \
    -d '{"target_column":"price","feature_columns":["volume"],"model_type":"linear"}' \
    "$BASE/api/excel/$SID/predict/$SHEET"
check "POST cluster"      200 -X POST -H "$J" -d '{"feature_columns":["price","volume"],"n_clusters":2}' \
    "$BASE/api/excel/$SID/cluster/$SHEET"
check "POST anomalies"    200 -X POST -H "$J" -d '{"feature_columns":["price","volume"],"method":"iqs"}' \
    "$BASE/api/excel/$SID/anomalies/$SHEET"
check "POST correlation"  200 -X POST -H "$J" -d '{"columns":["price","volume"]}' \
    "$BASE/api/excel/$SID/correlation/$SHEET"
check "GET  ml-recommendations" 200 "$BASE/api/excel/$SID/ml-recommendations/$SHEET"

# --- Batch (was a guaranteed 500) -------------------------------------------
BATCH="$(curl -s -m 120 -w $'\n__STATUS__%{http_code}' -F "files=@$TMP/fixture.xlsx" \
         -F "files=@$TMP/fixture.csv" "$BASE/api/excel/batch")"
BSTATUS="${BATCH##*__STATUS__}"; BBODY="${BATCH%$'\n'__STATUS__*}"
if [[ "$BSTATUS" == "200" ]] && grep -q '"success":true' <<<"$BBODY" && grep -q 'session_id' <<<"$BBODY"; then
    green "PASS  POST /api/excel/batch (200, returns session_ids)"
    PASS=$((PASS + 1))
else
    red "FAIL  POST /api/excel/batch (got $BSTATUS)"
    printf '      %s\n' "$(head -c 300 <<<"$BBODY")"
    FAIL=$((FAIL + 1))
fi

# --- RTOS -------------------------------------------------------------------
check "GET  rtos/status"       200 "$BASE/api/rtos/status"
check "POST rtos/boot"         200 -X POST "$BASE/api/rtos/boot"
check "POST rtos/watchdog"     200 -X POST -H "$J" -d '{"name":"system_watchdog"}' \
                              "$BASE/api/rtos/watchdog/feed"
check "POST rtos/enqueue"      200 -X POST -H "$J" \
    -d "{\"file_path\":\"$TMP/fixture.xlsx\",\"operations\":[{\"type\":\"remove_duplicates\"}]}" \
    "$BASE/api/rtos/excel/enqueue"
check "GET  rtos/results"      200 "$BASE/api/rtos/excel/results"

# --- Cloud (credentials optional) -------------------------------------------
ALLOW_BODY='error|success|available|configured' check "GET  cloud/status" 200 "$BASE/api/cloud/status"
ALLOW_BODY='error|success' check "POST cloud/upload" 200 -X POST -F "file=@$TMP/fixture.csv" \
    -F "provider=s3" -F "cloud_path=smoke.csv" "$BASE/api/cloud/upload"
ALLOW_BODY='error|success' check "GET  cloud/list" 200 "$BASE/api/cloud/list/s3"
ALLOW_BODY='error|success' check "GET  cloud/download" "200|400" "$BASE/api/cloud/download/s3?cloud_path=x.csv"

# --- Collaboration ----------------------------------------------------------
CID="$(python -c "import uuid;print(uuid.uuid4())")"
check "POST collaboration/sessions"   200 -X POST -H "$J" \
    -d "{\"user_id\":\"$CID\",\"session_name\":\"smoke\",\"file_info\":{\"name\":\"fixture.xlsx\"}}" \
    "$BASE/api/collaboration/sessions"
check "POST collaboration/join"       200 -X POST -H "$J" -d "{\"user_id\":\"$CID\"}" \
    "$BASE/api/collaboration/sessions/room-1/join"
check "GET  collaboration/state"      200 "$BASE/api/collaboration/sessions/room-1"
check "GET  collaboration/user"       200 "$BASE/api/collaboration/users/$CID/sessions"
check "POST collaboration/changes"    200 -X POST -H "$J" \
    -d '{"user_id":"'$CID'","change":{"type":"cell","cell":"A1","value":1}}' \
    "$BASE/api/collaboration/sessions/room-1/changes"
check "POST collaboration/leave"      200 -X POST -H "$J" -d "{\"user_id\":\"$CID\"}" \
    "$BASE/api/collaboration/sessions/room-1/leave"

# --- AI (disabled by default -> 503) ---------------------------------------
if [[ "${AI_ENABLED:-false}" == "true" ]]; then
    check "GET  ai/insights"                200 "$BASE/api/ai/insights/$SID/$SHEET"
    check "GET  ai/cleaning-suggestions"    200 "$BASE/api/ai/cleaning-suggestions/$SID/$SHEET"
    check "GET  ai/visualization-suggests"  200 "$BASE/api/ai/visualization-suggestions/$SID/$SHEET"
    check "POST ai/query"                   200 -X POST -H "$J" -d '{"query":"summarize this"}' \
                                         "$BASE/api/ai/query/$SID/$SHEET"
else
    check "GET  ai/insights -> 503 (AI off)" 503 "$BASE/api/ai/insights/$SID/$SHEET"
fi

# --- Validation (should reject, not 500) ------------------------------------
check "GET  bad session -> 400/404"   "400|404" -H "$J" "$BASE/api/excel/..%2F..%2Fetc/passwd/analyze/$SHEET"
check "POST predict bad format -> 400" 400 -X POST -H "$J" -d '{}' "$BASE/api/excel/$SID/predict/$SHEET"
check "GET  export bad format -> 400"  400 "$BASE/api/excel/$SID/export/$SHEET?format=bogus"

echo
echo "==============================="
echo " passed: $PASS   failed: $FAIL"
echo "==============================="
[[ "$FAIL" -eq 0 ]]
