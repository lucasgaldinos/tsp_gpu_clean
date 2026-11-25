# ==============================================================================
# .latexmkrc — LuaLaTeX + Biber build for second_draft.tex
#
# latexmk only reads the RC file in the *current working directory*, so run the
# commands below from documents_tcc/tex/thesis/ (or pass this file with -r).
#
#   latexmk second_draft.tex          # build second_draft.pdf
#   latexmk -pvc second_draft.tex     # watch sources and rebuild on save
#   latexmk -c second_draft.tex       # clean auxiliaries, keep second_draft.pdf
#   latexmk -C second_draft.tex       # full clean, also removes second_draft.pdf
#   latexmk -gg second_draft.tex      # full clean followed by a build
# ==============================================================================

# --- Engine: LuaLaTeX producing the PDF directly (same as `-lualatex`) --------
# 0 = none, 1 = pdflatex, 2 = ps->pdf, 3 = dvi->pdf, 4 = lualatex, 5 = xelatex.
$pdf_mode = 4;
# %O = options computed by latexmk (-interaction, -recorder, ...); %S = source.
$lualatex = 'lualatex %O %S';

# --- Bibliography: biblatex + biber (refs.bib) --------------------------------
# 2 = rerun biber whenever the .bcf changes; .bbl is treated as disposable.
$bibtex_use = 2;
# %B = job base name (second_draft).
$biber = 'biber %O %B';

# --- Reference/citation convergence -------------------------------------------
$max_repeat = 5;

# --- Default target -----------------------------------------------------------
# Without this, a bare `latexmk` compiles every *.tex in this directory,
# including the pandoc_support.tex fragment, which cannot be built standalone.
@default_files = ('second_draft');

# --- Cleanup ------------------------------------------------------------------
# latexmk already removes aux bcf blg fdb_latexmk fls idx ind lof log lot out
# toc xdv, plus bbl (because $bibtex_use = 2). These entries are the only
# project artefacts missing from that default set.
$clean_ext = 'run.xml';           # biber control file written by biblatex
$clean_full_ext = 'synctex.gz';   # final-output artefact: only -C removes it
