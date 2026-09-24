"""Neural execution heads for Geo-VLA (scene classifier + change detector).

Kept import-light (no torch) so the rest of the backend can run without it.
"""

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
