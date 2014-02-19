for f in *.mov; do ffmpeg -i "$f" -filter:v scale=320:-1 "${f%.mov}.mp4"; done
