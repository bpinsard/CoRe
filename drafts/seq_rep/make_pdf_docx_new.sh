#!/bin/sh
pandoc --filter pandoc-crossref --bibliography seq_rep_paper.bib -H ~/.pandoc/templates/fig_captions.tex --template=custom.latex -V geometry:margin=1.5cm -s seq_rep_new.md -o seq_rep_new.pdf

pandoc -t markdown --columns=2000 --template=custom.with_authors_affiliations.md -s seq_rep_new.md | sed -e "s/.pdf/.png/g; s/{#eq:.*}//g; s/\"\"//g;" | pandoc --filter pandoc-crossref --reference-docx=$HOME/.pandoc/templates/ref_margin_styles.docx  --bibliography seq_rep_paper.bib -o seq_rep_new.docx
