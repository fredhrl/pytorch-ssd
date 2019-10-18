"""
To run this script, you must have opencv installed in you python environment.
Type "python video_marker.py -h" for insctruction of how to run it.

Use the "C" key to start/stop registering a haircut.

Use the "B" key to start/stop registering a shaving.

Use the "O" key to slow down the video.

Use the "P" key to speed up the video.

When neither a "haircut" or a "shaving" is being resgistered, an "other" activity will be registered.

Look at the top-left of the screen to get info on the activity that is being tracked and the frame rate.
"""

import os
import pickle
import uuid
from argparse import ArgumentParser
import pandas as pd
import cv2
import numpy as np

main_wnd_name = "Video Marker"
trackbar_name = "trackbar"

eventD = []
out_file = []
flag  = 0
is_eventD = False


video_stopped = False

frame_count = 0
frame_interval = 136

argparser = ArgumentParser()
argparser.add_argument(
    "--video", "-v", help="The test video path", type=str, required=True
)
argparser.add_argument(
    "--outdir",
    "-o",
    help="The directory where the .pickle files will be located",
    type=str,
    required=False,
)
argparser.add_argument(
    "--namee",
    "-n",
    help="the namee of the file generated",
    type=str,
    required=True,
)


def process_intervals(intervals):
    new_intervals = []
    ignored_idxs = []

    for outter_idx, [start, end] in enumerate(intervals):
        if outter_idx in ignored_idxs:
            continue

        if start == end or end < start:
            ignored_idxs.append(outter_idx)
            continue

        for inner_idx, [inner_start, inner_end] in enumerate(intervals):
            if outter_idx == inner_idx or inner_idx in ignored_idxs:
                continue

            if inner_start == inner_end or inner_end < inner_start:
                ignored_idxs.append(inner_idx)
                continue

            if (
                inner_start <= start < inner_end
                or inner_start < end <= inner_end
                or start <= inner_start < end
                or start < inner_end <= end
            ):
                start = min(start, inner_start)
                end = max(end, inner_end)
                ignored_idxs.append(inner_idx)

        new_intervals.append([start, end])

    return new_intervals


def get_timeline_events():
    timeline = [[start, end, "eventD"] for [start, end] in eventD]

    timeline.sort(key=lambda event: event[0])

    count = 0
    while count + 1 < len(timeline):
        if timeline[count][1] > timeline[count + 1][0]:
            timeline[count + 1][0] = timeline[count][1]
        count += 1

    new_timeline = []
    last_frame = 0
    for [start, end, event_type] in timeline:
        if last_frame != start:
            new_timeline.append([last_frame, start, "other"])

        new_timeline.append([start, end, event_type])
        last_frame = end

    last_video_frame = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    if last_frame < last_video_frame:
        new_timeline.append([last_frame, last_video_frame, "other"])

    return new_timeline


def process_video(name):
    global eventD

    video_stream = cv2.VideoCapture(args.video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    eventD = process_intervals(eventD)

    timeline = get_timeline_events()

    counters = {"eventD": 0,"other":0}
    for [start, end, event] in timeline:
        counters[event] += 1
    
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, start)

        frame_count = 0
        [grabbed, frame] = video_stream.read()
        while grabbed and frame_count < (end - start):
            [grabbed, frame] = video_stream.read()
            frame_count += 1


def on_trackbar(val):
    global video_stopped
    global video_stream
    global frame
    global grabbed
    global frame_count
    global is_eventD

    if not video_stopped:
        return

    frame_count = cv2.getTrackbarPos(trackbar_name, main_wnd_name)

    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    (grabbed, frame) = video_stream.read()


if __name__ == "__main__":
    args = argparser.parse_args()

    if not os.path.isfile(args.video):
        raise SystemError("Invalid video file!")

    outdir = None

    if not args.outdir:
        outdir = os.path.splitext(args.video)[0]
    else:
        outdir = os.path.expanduser(args.outdir)
    eventName = args.namee
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    eventName = args.namee
    random_name = uuid.uuid4().hex

    eventD_file = os.path.join(outdir, random_name + "_eventD.pickle")
    if os.path.isfile(eventD_file):
        os.remove(eventD_file)

    video_stream = cv2.VideoCapture(args.video)

    cv2.namedWindow(main_wnd_name,cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        trackbar_name,
        main_wnd_name,
        0,
        int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
        on_trackbar,
    )

    (grabbed, frame) = video_stream.read()

    while grabbed:
        if not video_stopped:
            frame_count += 1
            cv2.setTrackbarPos(trackbar_name, main_wnd_name, frame_count)

        key = cv2.waitKey(int(frame_interval))
        if key == ord("q"):
            break
        elif key == ord("w"):
                flag = 1
                is_eventD = not is_eventD
                if is_eventD:
                    eventD.append([frame_count, None])
                else:
                    eventD[-1][1] = frame_count


        elif key == ord("o") and frame_interval < 136:
            frame_interval *= 2

        elif key == ord("p") and frame_interval > 17:
            frame_interval /= 2

        elif key == ord("s"):
            video_stopped = not video_stopped

        else:
            pass

        frame_copy = np.copy(frame)
        cv2.putText(
            frame_copy,
            f"is_eventD: {is_eventD}",
            (12, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
        )

        cv2.imshow(main_wnd_name, frame_copy)

        if not video_stopped:
            (grabbed, frame) = video_stream.read()

    if eventD and eventD[-1][1] is None:
        eventD[-1][1] = frame_count

    cv2.destroyAllWindows()

    if flag ==1:
        eventD_df = pd.DataFrame(eventD, columns= ["start","end"])
        eventD_df.to_csv(os.path.join(outdir, random_name + eventName + ".csv"))


    process_video(random_name)
