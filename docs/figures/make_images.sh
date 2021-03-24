#!/bin/bash
for file in $PWD/*.tex
do
    pdflatex $file
    b=$(basename $file)
    filename="${b%.*}"
    extension="${b##*.}"
    pdf2svg $filename.pdf $filename.svg
done

rm *.pdf
rm *.aux
rm *.out
rm *.log

