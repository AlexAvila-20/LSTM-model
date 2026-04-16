;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "titlepage" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "spanish" "es-nodecimaldot") ("inputenc" "utf8") ("float" "") ("listings" "") ("xcolor" "") ("graphicx" "") ("latexsym" "") ("amssymb" "") ("graphics" "") ("setspace" "") ("tcolorbox" "many") ("amsfonts" "") ("amsmath" "") ("physics" "") ("SIunits" "amssymb") ("geometry" "" "letterpaper" "bottom=25mm" "top=25mm") ("longtable" "") ("array" "") ("natbib" "") ("hyperref" "bookmarksnumbered" "breaklinks={true}" "pdfauthor={Bilbo Baggins (bilbo@theshire.middleearth)}" "pdftitle={El Hobbit o la historia de una ida y una vuelta}" "pdfsubject={Describir las aventuras de Bilbo}" "pdfkeywords={condensed matter materials \\& applied physics, quasiparticles \\&
  collective excitations, optical phonons}" "pdfauthor={Miguel Avila miguelavila.c@icloud.com}" "pdftitle={}" "pdftitle={Desarrollo de un modelo predictivo de precipitación para la región Bocacosta de Guatemala mediante redes neuronales LSTM}" "pdfsubject={Informe final de ejercicio profesional supervisado}" "pdfkeywords={meteorology, climate physics}")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "introducción"
    "objetivos"
    "marco_teórico"
    "metodología"
    "report"
    "rep11"
    "babel"
    "inputenc"
    "float"
    "listings"
    "xcolor"
    "graphicx"
    "latexsym"
    "amsmath"
    "amssymb"
    "graphics"
    "setspace"
    "tcolorbox"
    "geometry"
    "amsfonts"
    "physics"
    "SIunits"
    "longtable"
    "array"
    "natbib"
    "hyperref")
   (LaTeX-add-labels
    "sec:resumen"
    "sec:introduccion"
    "sec:objetivos"
    "sec:marco"
    "sec:metodologia"
    "sec:resultados"
    "sec:conclusiones"
    "sec:recomendaciones")
   (LaTeX-add-bibliographies
    "biblio"))
 :latex)

