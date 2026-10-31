#!/bin/bash
# usage: build.sh <audio_dir> <out.mp4>  — one continuous normalized audio track, video timed to it
set -e; A=$1; OUT=$2; T=$(mktemp -d); : > $T/a.txt; : > $T/v.txt
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  ffmpeg -loglevel error -y -i $A/$i.wav -af "silenceremove=stop_periods=-1:stop_duration=0.45:stop_threshold=-45dB:stop_silence=0.3,loudnorm=I=-16:TP=-1.5:LRA=7,apad=pad_dur=0.8,adelay=500|500" -ar 44100 -ac 1 $T/$i.wav
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $T/$i.wav)
  echo "file $T/$i.wav" >> $T/a.txt; printf "file '%s'\nduration %s\n" "$PWD/frames/$i.png" "$d" >> $T/v.txt
done
echo "file '$PWD/frames/11.png'" >> $T/v.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i $T/a.txt -c:a pcm_s16le $T/all.wav
ffmpeg -loglevel error -y -f concat -safe 0 -i $T/v.txt -i $T/all.wav -vf "fps=25,scale=1920:1080,format=yuv420p" -c:v libx264 -c:a aac -b:a 160k -shortest -movflags +faststart $OUT
cp $T/all.wav ${OUT%.mp4}.wav; rm -rf $T
