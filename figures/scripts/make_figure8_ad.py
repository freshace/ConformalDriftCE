"""Render Figure 8 with panels a-d only, preserving their original geometry.

Run: python paper/2026-10-23/make_figure8_ad.py
Optional alignment QA: --qa-scripts path/to/nature-figure/scripts
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "paper/2026-10-18/manuscript_cn/figures_en/scripts/make_figures_en.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path)
    args = parser.parse_args()
    qa = HERE / "figure8_ad_qa"
    qa.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(qa / "mplcache")
    spec = importlib.util.spec_from_file_location("english_figures", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def save_ad(fig, _name):
        from matplotlib.transforms import Bbox

        assert len(fig.axes) == 6
        for ax in list(fig.axes[4:]):
            fig.delaxes(ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Remove the unused bottom of the canvas, keeping a 0.08-inch margin
        # below the retained x-axis labels. No rescaling or relayout of a-d.
        bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes) / fig.dpi - .08
        width, height = fig.get_size_inches()
        export_box = Bbox.from_extents(0, bottom, width, height)
        if args.qa_scripts:
            sys.path.insert(0, str(args.qa_scripts))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(
                fig, json_out=qa / "alignment.json", strict=True,
            )
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(HERE / f"figure8_ad.{ext}", dpi=600, facecolor="white",
                        bbox_inches=export_box, pad_inches=0)
        (qa / "layout.json").write_text(json.dumps({
            "panels": ["a", "b", "c", "d"],
            "width_inches": float(width),
            "height_inches": float(height - bottom),
            "removed_bottom_inches": float(bottom),
            "retained_panels_rescaled": False,
            "data_and_plotting_logic": "Original fig09_continuous_repair()",
        }, indent=2), encoding="utf-8")
        module.plt.close(fig)
        print(f"Saved {HERE / 'figure8_ad.png'}")

    module.save = save_ad
    module.fig09_continuous_repair()


if __name__ == "__main__":
    main()
