all:
	pdflatex multi-dimensional-params.tex
	pdflatex one-dimensional-state-observation.tex
	pdflatex one-dimensional-state.tex

	convert -density 500 -quality 100 multi-dimensional-params.pdf -background white -flatten multi-dimensional-params.png
	convert -density 500 -quality 100 one-dimensional-state-observation.pdf -background white -flatten one-dimensional-state-observation.png
	convert -density 500 -quality 100 one-dimensional-state.pdf -background white -flatten one-dimensional-state.png
