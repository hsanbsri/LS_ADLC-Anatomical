"""make_figures.py: regenerates Figs. 1-6 of the paper in the journal format.

Runs the six figure scripts unchanged in data, layout and selection logic and
only adapts the typography to the journal's artwork guidelines:
  * font: Liberation Sans (metric-compatible with Arial), mathtext in the same family;
  * all font sizes multiplied by a per-figure factor so that lettering is
    8-12 pt at the printed width (174 mm);
  * output: figures/output/Fig1..Fig6.{pdf,png,eps} (PNG at 600 dpi, RGB; EPS needs
    pdftops from poppler-utils), numbered in order of appearance in the paper.
Usage: python figures/make_figures.py [N ...]     (default: all six)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# (figure number, script, output stem written by the script, font-size factor)
FIGS = [
    (1, "fig1_examples.py", "fig_examples", 1.30),
    (2, "fig2_overview.py", "fig1_overview", 1.25),
    (3, "fig3_backbone.py", "fig1_backbone", 1.10),
    (4, "fig4_gate.py", "fig_gate", 1.15),
    (5, "fig5_replication.py", "fig5_replication", 1.00),
    (6, "fig6_per_class.py", "fig6_per_class_bar", 1.10),
]

_orig_set_size = fm.FontProperties.set_size


def run(num, script, stem, factor):
    import matplotlib.pyplot as plt
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams["font.size"] = matplotlib.rcParamsDefault["font.size"] * factor

    def set_size(self, size):                       # scale every explicit numeric size once
        if isinstance(size, (int, float)):
            size = size * factor
        _orig_set_size(self, size)
    fm.FontProperties.set_size = set_size

    path = os.path.join(HERE, script)
    src = open(path).read()
    src = src.replace('"DejaVu Sans"', '"Liberation Sans"')
    src = src.replace('"dejavusans"', '"custom"')
    src = src.replace('plt.rcParams["font.family"] = "Liberation Sans"',
                      'plt.rcParams["font.family"] = "Liberation Sans"\n'
                      'plt.rcParams["mathtext.fontset"] = "custom"\n'
                      'plt.rcParams["mathtext.rm"] = "Liberation Sans"\n'
                      'plt.rcParams["mathtext.it"] = "Liberation Sans:italic"\n'
                      'plt.rcParams["mathtext.bf"] = "Liberation Sans:bold"\n'
                      'plt.rcParams["mathtext.fallback"] = "stixsans"\n'
                      'plt.rcParams["pdf.fonttype"] = 42')
    # output name FigN, numbered in order of appearance
    src = src.replace(f'"{stem}.png"', f'"Fig{num}.png"').replace(f'"{stem}.pdf"', f'"Fig{num}.pdf"')
    src = src.replace(f'f"{stem}.{{ext}}"', f'f"Fig{num}.{{ext}}"')
    if "Liberation Sans" not in src:                     # scripts without an explicit family
        src = 'import matplotlib.pyplot as plt\nplt.rcParams["font.family"] = "Liberation Sans"\n' \
              'plt.rcParams["pdf.fonttype"] = 42\n' + src
    if num == 6:                                     # figure parts in lowercase letters
        src = src.replace('"A. Per-class F1', '"a  Per-class F1').replace('"B. SPSW-F1 per', '"b  SPSW-F1 per')
        assert '"a  Per-class F1' in src and '"b  SPSW-F1 per' in src
    assert f"Fig{num}" in src, f"output rename failed for {script}"
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)
    fm.FontProperties.set_size = _orig_set_size
    from PIL import Image                           # RGB, 8 bits per channel (no alpha)
    png = os.path.join(OUT, f"Fig{num}.png"); im = Image.open(png)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        bg.save(png, dpi=im.info.get("dpi", (600, 600)))
    os.system(f"pdftops -eps -level3 {os.path.join(OUT, f'Fig{num}.pdf')} {os.path.join(OUT, f'Fig{num}.eps')}")
    print(f"Fig{num} <- {script} (font x{factor})")


if __name__ == "__main__":
    want = {int(a) for a in sys.argv[1:]} or {f[0] for f in FIGS}
    for f in FIGS:
        if f[0] in want:
            run(*f)
