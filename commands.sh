for f in *.mov; do ffmpeg -i "$f" -filter:v scale=320:-1 "${f%.mov}.mp4"; done


for f in *.mp4; do ffmpeg -i "$f" -ss 60 -t 10 "${f%.mp4}_10secs.mp4"; done