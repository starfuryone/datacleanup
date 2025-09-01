try:
    from prometheus_client import Counter, Gauge
except Exception:  # fallback if not installed
    class _Noop:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def set(self, *_): pass
    def Counter(*_, **__): return _Noop()
    def Gauge(*_, **__): return _Noop()

REQUESTS = Counter("requests_total", "Total API requests", ["route"])
DATA_ROWS = Gauge("data_rows", "Rows processed")

def increment_counter(name: str, labels: dict | None = None):
    if name == "requests_total":
        (REQUESTS.labels(**(labels or {}))).inc()

def set_gauge(name: str, value: float):
    if name == "data_rows":
        DATA_ROWS.set(value)
