all: worksheet_linearRegression.pdf

%.pdf: %.tex
	pdflatex $<*
	bibtex $*.aux
	pdflatex $<*
	pdflatex $<*

