from pdflatex import PDFLaTeX

pdfl = PDFLaTeX.from_texfile('my_arch.tex')
pdfl.set_interaction_mode()
pdf = pdfl.create_pdf()
