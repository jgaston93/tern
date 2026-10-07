set -e
${PYTHON:-python3} - <<'PY'
from pkg.strings import shout
from pkg.numbers import double
from pkg.lists import tail
assert shout("hi") == "HI!", shout("hi")
assert double(21) == 42, double(21)
assert tail([1, 2, 3]) == [2, 3], tail([1, 2, 3])
print("ok")
PY
