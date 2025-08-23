#!/bin/bash
# Create 30-minute test video by concatenating pilot videos

echo "Creating 30-minute test video for compliance testing..."

# Create file list for concatenation (excluding very large files)
echo "# Pilot videos concatenation list" > /tmp/concat_list.txt

# Add each pilot video (excluding large ones)
for video in assets/pilots/{art_painting,beach_sunset,city_aerial,cooking_kitchen,mountain_climb,nature_forest,office_work,sports_soccer,tech_coding,urban_traffic}.mp4; do
    if [ -f "$video" ]; then
        echo "file '$PWD/$video'" >> /tmp/concat_list.txt
    fi
done

# DON'T repeat - videos are already 3 minutes each, 10 videos = 30 minutes total
# No need to repeat since each video is 180 seconds

echo "Concatenation list created with $(grep -c '^file' /tmp/concat_list.txt) segments"

# Check approximate duration
echo "Checking input video durations..."
total_duration=0
for video in assets/pilots/{art_painting,beach_sunset,city_aerial,cooking_kitchen,mountain_climb,nature_forest,office_work,sports_soccer,tech_coding,urban_traffic}.mp4; do
    if [ -f "$video" ]; then
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null)
        echo "  $(basename $video): ${duration}s"
        total_duration=$(echo "$total_duration + $duration" | bc)
    fi
done

echo "Total input duration: ${total_duration}s (~$(echo "scale=1; $total_duration / 60" | bc) minutes)"

# Create the concatenated video
echo "Creating concatenated video..."
ffmpeg -f concat -safe 0 -i /tmp/concat_list.txt \
    -c:v libx264 -preset fast -crf 23 \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    -y assets/test_30min.mp4

# Verify the output
if [ -f "assets/test_30min.mp4" ]; then
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 assets/test_30min.mp4)
    size=$(du -h assets/test_30min.mp4 | cut -f1)
    
    echo ""
    echo "✅ 30-minute test video created successfully!"
    echo "  Path: assets/test_30min.mp4"
    echo "  Duration: ${duration}s ($(echo "scale=1; $duration / 60" | bc) minutes)"
    echo "  Size: $size"
    
    # Also get video properties
    echo ""
    echo "Video properties:"
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,bit_rate -of default=noprint_wrappers=1 assets/test_30min.mp4
else
    echo "❌ Failed to create test video!"
    exit 1
fi

# Clean up
rm -f /tmp/concat_list.txt

echo ""
echo "Test video ready for Day 2 memory and performance testing!"