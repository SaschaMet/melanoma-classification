#!/bin/bash
rm -rf tmp

pdflatex -synctex=1 document
bibtex document
pdflatex -synctex=1 document
pdflatex -synctex=1 document

# move temporary files to tmp folder
mkdir tmp
mv *.aux tmp
mv *.bbl tmp
mv *.blg tmp
mv *.lof tmp
mv *.log tmp
mv *.lol tmp
mv *.lot tmp
mv *.out tmp
mv *.toc tmp
mv chapters/*.aux tmp
mv frontmatter/*.aux tmp
mv appendix/*.aux tmp

rm -rf tmp