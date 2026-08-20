#!/usr/bin/env bash
# Run the unit test suite.
#
#   ./tests/run_all.sh            all tests
#   ./tests/run_all.sh -v         verbose (per-test names)
#
# Nothing here talks to a network, a model server, a GPU or a real Redis. Every test uses
# in-memory SQLite and stubbed endpoints. A test that needed real hardware would be a benchmark,
# and those live in scripts/ and write their results to bench/.
#
# The services import PyYAML lazily, inside the functions that read a config file, and no test
# reaches those paths — so the whole suite runs on plain system python3 as well as in the
# services venv. It prefers the venv when present because that is the interpreter the services
# actually run under, and a difference between the two is worth catching here rather than in
# production.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VERBOSE=()
[ "${1:-}" = "-v" ] && VERBOSE=(-v)

TESTS=(tests/test_rules.py         # compliance state machine
       tests/test_events.py        # event vocabulary and transition detection
       tests/test_store.py         # incident state machine: refcounting, merging, re-raising
       tests/test_agent.py         # planner vocabulary, tools, schema bounds, confidence
       tests/test_reasoning.py     # verdict computation and hazard escalation
       tests/test_clip_service.py  # clip windowing, PTS arithmetic, retention
       tests/test_notify.py)       # notification policy

VENV="${ROOT}/build/venv-services/bin/python3"
if [ -x "$VENV" ]; then
  PY="$VENV";      WHICH="build/venv-services"
else
  PY="$(command -v python3)"; WHICH="system python3 (build/venv-services not found)"
fi
echo "interpreter: ${WHICH}"

PASSED=0; FAILED=()
for t in "${TESTS[@]}"; do
  printf "\n\033[1m── %s\033[0m\n" "$t"
  if "$PY" "$t" "${VERBOSE[@]+"${VERBOSE[@]}"}"; then
    PASSED=$((PASSED + 1))
  else
    FAILED+=("$t")
  fi
done

echo
echo "──────────────────────────────────────────────"
printf " %d of %d test files passed\n" "$PASSED" "${#TESTS[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf " failed: %s\n" "${FAILED[*]}"
  echo "──────────────────────────────────────────────"
  exit 1
fi
echo "──────────────────────────────────────────────"
