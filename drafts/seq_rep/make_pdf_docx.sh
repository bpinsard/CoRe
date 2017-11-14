#!/bin/sh
pandoc  --filter pandoc-fignos --reference-docx=seq_rep_ref_margin.docx --bibliography seq_rep_paper.bib -s seq_rep.md -o seq_rep.docx
pandoc --filter pandoc-fignos --template=custom.latex -V geometry:margin=1.2cm --bibliography seq_rep_paper.bib -s seq_rep.md -o seq_rep.pdf
