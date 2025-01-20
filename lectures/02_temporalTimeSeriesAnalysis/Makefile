all: temporalTimeSeriesAnalysis.pdf

%.pdf: *.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
