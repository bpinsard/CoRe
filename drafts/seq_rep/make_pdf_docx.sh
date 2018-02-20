#!/bin/sh
pandoc  --filter pandoc-crossref --reference-docx=seq_rep_ref_margin.docx --bibliography seq_rep_paper.bib -s seq_rep.md -o seq_rep.docx
pandoc --filter pandoc-crossref -H ~/.pandoc/templates/fig_captions.tex --template=custom.latex -V geometry:margin=1.5cm --bibliography seq_rep_paper.bib  -V lineno=1 -s seq_rep.md -o seq_rep.pdf
