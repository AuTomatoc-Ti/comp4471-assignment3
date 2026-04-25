import argparse
import os
import subprocess

try:
    from PyPDF2 import PdfMerger

    MERGE = True
except ImportError:
    print("Could not find PyPDF2. Leaving pdf files unmerged.")
    MERGE = False


def _sanitize_tex(tex_path):
    """Remove legacy inputenc/ucs settings that break modern XeLaTeX."""
    with open(tex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        if "\\usepackage[mathletters]{ucs}" in line:
            continue
        if "\\usepackage[utf8x]{inputenc}" in line:
            continue
        filtered.append(line)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.writelines(filtered)


def _build_pdf_from_notebook(notebook_file):
    base = os.path.splitext(notebook_file)[0]
    tex_file = base + ".tex"
    pdf_file = base + ".pdf"

    # 1) Export notebook to LaTeX
    subprocess.run(
        ["jupyter", "nbconvert", "--log-level", "CRITICAL", "--to", "latex", notebook_file],
        check=True,
    )

    # 2) Patch TeX for current TeX Live compatibility
    _sanitize_tex(tex_file)

    # 3) Compile with xelatex
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_file],
        check=True,
    )

    # A second pass helps resolve references/table of contents.
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_file],
        check=True,
    )

    # Clean temporary LaTeX artifacts
    for ext in (".aux", ".log", ".out", ".toc", ".tex"):
        tmp = base + ext
        if os.path.exists(tmp):
            os.remove(tmp)

    return pdf_file


def main(files, pdf_name):
    pdfs = []
    for f in files:
        pdf_path = _build_pdf_from_notebook(f)
        pdfs.append(pdf_path)
        print("Created PDF {}.".format(f))

    if MERGE:
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(pdf_name)
        merger.close()
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We pass in a explicit notebook arg so that we can provide an ordered list
    # and produce an ordered PDF.
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename)
