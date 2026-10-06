set -e
! grep -rn "calc_total" --include=*.py .
${PYTHON:-python3} - <<'PY'
from app.pricing import compute_order_total
from app.checkout import checkout
from app.report import daily_report
assert checkout([(10.0, 2)], member=True)["total"] == 19.44
assert daily_report([[(10.0, 1)]])["revenue"] == 10.8
PY
