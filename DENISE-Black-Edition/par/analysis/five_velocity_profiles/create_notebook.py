from pathlib import Path

import nbformat as nbf

out = Path("/home/ywb/DENISE-Black-Edition/par/analysis/five_velocity_profiles")
nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Five-position Marmousi Vp comparison\n\n"
        "## tl;dr\n\nThis notebook extracts five vertical Vp profiles and compares the initial, true, and final inverted models."
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "The model is 500 × 174 with 20 m spacing. DENISE stores y contiguously inside each x column. "
        "Profiles are selected uniformly at x = 1, 3, 5, 7, and 9 km."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
        "OUT = Path('/home/ywb/DENISE-Black-Edition/par/analysis/five_velocity_profiles')\n"
        "%run /home/ywb/DENISE-Black-Edition/par/analysis/five_velocity_profiles/analyze_profiles.py"
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        "profiles = pd.read_csv(OUT / 'five_vp_profiles.csv')\n"
        "summary = pd.read_csv(OUT / 'profile_error_summary.csv')\n"
        "profiles.head(), profiles.shape"
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell("summary.round(3)"),
    nbf.v4.new_code_cell("display(Image(filename=OUT / 'five_vp_profiles.png'))"),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\nCompare inverted RMSE/MAE with the initial-model baseline at each x position. "
        "The full depth-by-depth values are saved in `five_vp_profiles.csv`."
    ),
]
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, out / "five_velocity_profiles.ipynb")
