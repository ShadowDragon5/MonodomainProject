filename=Project

pdf:
	@pdflatex -shell-escape ${filename}
	@bibtex ${filename}
	@pdflatex -shell-escape ${filename} # for refs
	@pdflatex -shell-escape ${filename} # for refs

clean:
	@rm -f ${filename}.out ${filename}.log ${filename}.bbl ${filename}.blg ${filename}.au ${filename}.log ${filename}.ps ${filename}.aux ${filename}.out ${filename}.dvi ${filename}.bbl ${filename}.blg ${filename}.toc ${filename}.nav ${filename}.vrb ${filename}.snm ${filename}.bcf ${filename}.run.xml ${filename}.lot ${filename}.lof
