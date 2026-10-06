set -e
${PYTHON:-python3} - <<'PY'
from stats import median, mean, spread
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
assert median([1, 2]) == 1.5
assert mean([1, 2, 3]) == 2 and spread([1, 5]) == 4
PY
