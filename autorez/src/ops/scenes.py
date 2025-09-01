import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_scenes_from_video(video_path: str, threshold: float = 0.25) -> list[float]:
    """
    Detects scene cuts in a video based on histogram differences, as specified
    in Blueprint Section 9.

    Args:
        video_path: Path to the video file.
        threshold: L1 distance threshold for detecting a cut.

    Returns:
        A list of timestamps (in seconds) where scene cuts were detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.warning("Video FPS is 0, cannot calculate cut timestamps.")
        return []

    prev_hist = None
    cuts = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process 1 frame per second as per blueprint
        if frame_count % int(fps) != 0:
            continue

        # Convert to RGB and calculate histogram
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()

        if prev_hist is not None:
            # Calculate L1 distance
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) # Using Bhattacharyya distance as a robust metric
            l1_dist = np.linalg.norm(prev_hist - hist, ord=1) / 2 # Normalized L1 distance

            if l1_dist > threshold:
                timestamp = frame_count / fps
                cuts.append(timestamp)
                logger.info(f"Scene cut detected at {timestamp:.2f}s (L1 diff: {l1_dist:.2f})")

        prev_hist = hist

    cap.release()
    return cuts
