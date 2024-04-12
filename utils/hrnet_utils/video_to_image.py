import cv2
import time
import os
import shutil

curDir = os.path.dirname(__file__)


def video_to_frames(input_loc, output_loc):
    if os.path.isdir(output_loc):
        # os.removedirs(output_loc)
        shutil.rmtree(output_loc)
        print(f"removed {output_loc}")
    try:
        os.makedirs(output_loc)
    except OSError:
        print("couldn't make it")
        pass

    time_start = time.time()
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")

    file_names = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(output_loc + "/%#05d.png" % (count + 1), frame)
        file_names.append(str(count + 1).zfill(5))
        count = count + 1
        if count > (video_length - 1):
            time_end = time.time()
            cap.release()
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds reconversion." % (time_end - time_start))
            break

    with open('output/test_indices.txt', 'w') as f:
        f.write(','.join(file_names))


if __name__ == "__main__":
    inputLoc = f'{curDir}/video/traffic.mp4'
    outputLoc = f'{curDir}/output/frames/traffic/'
    video_to_frames(inputLoc, outputLoc)
