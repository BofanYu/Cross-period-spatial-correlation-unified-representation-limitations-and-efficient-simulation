"""CSV output for experiment settings and numerical checks."""
import csv


def write_metadata(path, values):
    def flatten(value, key=""):
        if isinstance(value, dict):
            for name, item in value.items():
                yield from flatten(item, f"{key}.{name}" if key else str(name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from flatten(item, f"{key}.{index}")
        else:
            yield key, value

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        writer.writerows(flatten(values))
