"""Pre-download the network resources feather fetches lazily at run time.

Run at image build time so diagnostics work on compute nodes without
internet access:

* cartopy Natural Earth shapes behind ``ax.coastlines()`` and
  ``cfeature.LAND``/``COASTLINE`` (110m default, 50m for finer maps)
* regionmask's AR6 reference-region polygons (``feather/util/ar6_regions.py``)
"""

import os
from pathlib import Path

import cartopy
import cartopy.io.shapereader as shpreader
import regionmask

# At run time CARTOPY_DATA_DIR is cartopy's read-only ``pre_existing_data_dir``;
# downloads go to ``data_dir``, so point that at the same place here.
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["data_dir"] = Path(os.environ["CARTOPY_DATA_DIR"])

for scale in ("110m", "50m"):
    for category, name in (("physical", "coastline"), ("physical", "land"),
                           ("physical", "ocean")):
        path = shpreader.natural_earth(resolution=scale, category=category, name=name)
        print(f"cartopy {scale} {name}: {path}")

ar6 = regionmask.defined_regions.ar6.all
print(f"regionmask AR6: {len(ar6)} regions")
