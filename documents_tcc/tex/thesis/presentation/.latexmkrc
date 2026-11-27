# Configuração do latexmk para compilação da apresentação Beamer em LuaLaTeX + Biber.
# Uso: cd documents_tcc/tex/thesis/presentation && latexmk presentation.tex

$pdf_mode = 4;                  # 4 = LuaLaTeX
$lualatex = 'lualatex %O %S';
$bibtex_use = 2;                # Sempre executa biber/bibtex quando .bcf/.aux muda
$biber = 'biber %O %B';
$max_repeat = 5;

# Limpeza segura com `latexmk -c` (mantém presentation.pdf e fontes .tex)
$clean_ext = 'nav snm vrb run.xml bbl bcf blg fls fdb_latexmk synctex.gz';
