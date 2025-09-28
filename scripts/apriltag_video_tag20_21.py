#!/usr/bin/env python

from argparse import ArgumentParser
import os
import cv2
import numpy as np
import apriltag

################################################################################

def apriltag_video(input_streams=[0],
                   output_stream=False,
                   display_stream=True,
                   detection_window_name='AprilTag'):
    """
    Detect ONLY tags 20 and 21 from video. If both are visible,
    draw a line between their image centers and display:
      - pixel distance between centers
      - 3D distance between tag centers (in meters), using pose estimation
    """

    parser = ArgumentParser(description='Detect AprilTags from video stream.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    # Set up a reasonable search path for the apriltag DLL/so/dylib.
    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    # Hard-coded camera intrinsics and tag size (in meters)
    # Adjust to your camera for accurate 3D distance.
    camera_params = (1456.6855555383966, 1452.2059008627816, 966.8814607344126, 549.333443447071)
  # fx, fy, cx, cy
    tag_size_m = 0.049  # meters

    want_ids = {20, 21}

    for stream in input_streams:
        video = cv2.VideoCapture(stream)

        output = None
        if output_stream:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
            codec = cv2.VideoWriter_fourcc(*'XVID')
            if type(stream) != int:
                output_path = os.path.join('..', 'media', 'output', os.path.splitext(os.path.basename(str(stream)))[0] + '.avi')
            else:
                output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output = cv2.VideoWriter(output_path, codec, fps, (width, height))

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Run detection (no verbose prints, no built-in annotations)
            result, overlay = apriltag.detect_tags(
                frame,
                detector,
                camera_params=camera_params,
                tag_size=tag_size_m,
                vizualization=0,   # keep the overlay clean; we'll draw our own
                verbose=0,
                annotation=False
            )

            # Collect centers and poses for tag 20 and 21, if found
            centers = {}
            tvecs = {}
            for i in range(0, len(result), 4):
                det = result[i]        # apriltag.Detection
                pose = result[i + 1]   # 3x4 pose matrix (tag->camera)
                tid = getattr(det, 'tag_id', None)
                if tid in want_ids:
                    centers[tid] = np.array(det.center, dtype=float)
                    tvecs[tid] = np.array(pose[:3, 3], dtype=float)

            # If both present, compute distances and draw
            if 20 in centers and 21 in centers:
                # 2D pixel distance
                pix_dist = float(np.linalg.norm(centers[20] - centers[21]))

                # 3D distance in meters (using tag poses in camera frame)
                if 20 in tvecs and 21 in tvecs:
                    m_dist = float(np.linalg.norm(tvecs[20] - tvecs[21]))
                else:
                    m_dist = None

                # Draw a line between the tag centers
                p0 = tuple(np.round(centers[20]).astype(int))
                p1 = tuple(np.round(centers[21]).astype(int))
                cv2.line(overlay, p0, p1, (0, 255, 255), 2)

                # Prepare the text
                if m_dist is not None:
                    text = f"20↔21: {pix_dist:.1f}px | {m_dist:.3f} m"
                else:
                    text = f"20↔21: {pix_dist:.1f}px"

                # Put the text near the midpoint of the segment
                mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
                cv2.putText(overlay, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Also print to terminal
                print(text)

            # Optionally write/display
            if output_stream and output is not None:
                output.write(overlay)

            if display_stream:
                cv2.imshow(detection_window_name, overlay)
                # Press space bar to terminate
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

        if output_stream and output is not None:
            output.release()
        video.release()

################################################################################

if __name__ == '__main__':
    apriltag_video()


