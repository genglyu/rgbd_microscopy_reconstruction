import numpy
import cv2
import time
import os.path

# cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_EXPOSURE,-12)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,-2)
cap.set(cv2.CAP_PROP_TEMPERATURE,0)
cap.set(cv2.CAP_PROP_TEMPERATURE,0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, False)
cap.set(cv2.CAP_PROP_AUTO_WB, False)
# cap.set(cv2.CAP_PROP_SETTINGS, True)


# print(cv2.CAP_PROP_SETTINGS)


time_step = 1

file_dir = "./images/"
filename_template = "tile_%01d_%06d.png"


capturing = False
zoom_level = 1
image_count = 0
previous_time = time.time()

ret, frame = cap.read()
p_frame = frame
pp_frame = p_frame

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(capturing)

    current_time = time.time()
    if current_time - previous_time > time_step and capturing:
        filename = filename_template % (zoom_level, image_count)

        print("save as " + filename)

        cv2.imwrite(os.path.join(file_dir ,filename), frame)

        previous_time = current_time
        pp_frame = p_frame
        p_frame = frame
        image_count += 1
        capturing = False

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    side_by_side = numpy.concatenate((pp_frame, p_frame, frame), axis=1)

    # Display the resulting frame
    cv2.imshow('frame', side_by_side)

    key = cv2.waitKey(1)
    if key == ord('c'):
        capturing = True
    elif key == ord('v'):
        capturing = False
    elif key == ord('b'):
        if image_count > 0:
            image_count = image_count - 1
            print("Goes back to tile %6d" % (image_count))
        if image_count >= 1:
            p_frame = cv2.imread(os.path.join(file_dir, filename_template) % (zoom_level, image_count - 1))
        if image_count >= 2:
            pp_frame = cv2.imread(os.path.join(file_dir, filename_template) % (zoom_level, image_count - 2))

    elif key == ord('0'):
        zoom_level = 0
        print("switch to zoom level 0. Filename should be like tile_0_***.png")
    elif key == ord('1'):
        zoom_level = 1
        print("switch to zoom level 1. Filename should be like tile_1_***.png")
    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()