TEX=writeup.tex
BIB=*.bib

# Core compilation commands
CC=pdflatex
BB=bibtex

all: $(TEX)
	$(CC) $(TEX)
	rm -rf *.log *.aux *.bbl *.blg *.out *~
    
clean:
	rm -rf *.log *.aux *.bbl *.blg *.out *~
