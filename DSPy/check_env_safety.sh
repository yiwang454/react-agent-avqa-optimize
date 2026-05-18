python - <<'PY'
import site, sys
paths = []
for fn in (getattr(site, "getsitepackages", lambda: [])(), [site.getusersitepackages()]):
    for p in fn if isinstance(fn, list) else []:
        paths.append(p)
print("\n".join(paths))
PY