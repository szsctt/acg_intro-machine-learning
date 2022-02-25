FILES="*.svg"


for f in $FILES; do
 filename="${f%.*}"
 echo $filename
 magick $f "${filename}.png"
done