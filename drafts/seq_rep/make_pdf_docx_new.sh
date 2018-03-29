#!/bin/sh
pandoc  --filter pandoc-crossref --bibliography seq_rep_paper.bib --csl vancouver-author-date.csl -H ~/.pandoc/templates/fig_captions.tex --template=custom.latex -V lineno=1 -s seq_rep_new.md -o seq_rep_new.pdf

pandoc -t markdown --atx-headers --columns=2000 --template=custom.with_authors_affiliations.md -s seq_rep_new.md | sed -e "s/.pdf/.png/g; s/{#eq:.*}//g; s/\"\"//g;" | pandoc --filter pandoc-crossref --reference-docx=$HOME/.pandoc/templates/ref_margin_styles.docx  --bibliography seq_rep_paper.bib --csl vancouver-author-date.csl -M chapDelim:'' -o seq_rep_new.docx
