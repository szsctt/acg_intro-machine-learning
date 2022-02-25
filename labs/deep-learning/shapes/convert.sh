FILES="*.svg"


for f in $FILES; do
 filename="${f%.*}"
 magick $f "${filename}.png"
done