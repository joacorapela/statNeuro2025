all: spectralTimeSeriesAnalysis.pdf

%.pdf: *.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
