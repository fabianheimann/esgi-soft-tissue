#! /bin/sh

ffmpeg -r 25 -f image2 -s 2447x1335 -i sequenceM/frame_%03d.png -vcodec libx264 -qp 0 -preset veryslow test.mp4
