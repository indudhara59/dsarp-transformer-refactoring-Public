"""Project-wide constants."""

SELECTED_SMELLS = ("godComponent", "unstableDep", "hubLikeDep")
INPUT_FILENAMES = {
    "affects": "affects-smell-affects.csv",
    "metrics": "metrics-component-metrics.csv",
    "smells": "smells-smell-characteristics.csv",
}
COMMON_KEYS = ["project", "version_id"]

