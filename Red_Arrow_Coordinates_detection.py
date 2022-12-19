import cv2 as cv
import numpy as np
import math


def nothing(x):
    # any operation
    pass


cap = cv.VideoCapture(0)

cv.namedWindow("Trackbars")
cv.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv.FONT_HERSHEY_COMPLEX

# Create two variables to store the two instances of the arrow. Initially, set them to None.
original = None
final = None

while True:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("L-H", "Trackbars")
    l_s = cv.getTrackbarPos("L-S", "Trackbars")
    l_v = cv.getTrackbarPos("L-V", "Trackbars")
    u_h = cv.getTrackbarPos("U-H", "Trackbars")
    u_s = cv.getTrackbarPos("U-S", "Trackbars")
    u_v = cv.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel)

    # Contours detection
    if int(cv.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_AP)
    for cnt in contours:
        area=cv.contourArea(cnt)
        approx=cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
        if area > 1000:
            cv.drawContours(frame, [approx], 0, (0, 0, 0), 5)

    # Check if the key 'O' is pressed. If it is, set `original` to the current frame and the approx polygon of the arrow.
    if cv.waitKey(1) & 0xFF == ord('O'):
        original=(frame, approx)
        print("Original image captured")

    # Check if the key 'F' is pressed. If it is, set `final` to the current frame and the approx polygon of the arrow.
    if cv.waitKey(1) & 0xFF == ord('F'):
        final=(frame, approx)
        print("Final image captured")

    # Check if the Enter key is pressed. If it is, calculate the angle of rotation between `original` and `final` using the approach described in the previous answer.
    if cv.waitKey(1) & 0xFF == 13:
        if original and final:
            # Find the center of the arrow by taking the mean of the x and y coordinates of the points in the approx polygon
            original_center_x=sum(point[0]
                                  for point in original[1]) / len(original[1])
            original_center_y=sum(point[1]
                                  for point in original[1]) / len(original[1])
            original_center=(original_center_x, original_center_y)

            final_center_x=sum(point[0] for point in final[1]) / len(final[1])
            final_center_y=sum(point[1] for point in final[1]) / len(final[1])
            final_center=(final_center_x, final_center_y)

            # Calculate the angle between the center of the arrow and one of the points in the approx polygon using the arctan2 function
            original_angle_with_x_axis=math.atan2(
                original_center[1] - original[1][0][1], original_center[0] - original[1][0][0])
            final_angle_with_x_axis=math.atan2(
                final_center[1] - final[1][0][1], final_center[0] - final[1][0][0])

            # Calculate the angle between the original location of the arrow and the current location using the arctan2 function and the difference between the x and y coordinates of the two locations
            original_angle_with_final_location=math.atan2(
                final_center_y - original_center_y, final_center_x - original_center_x)

            # Subtract the angle between the center of the arrow and one of the points in the approx polygon from the angle between the original location and the current location to get the angle of rotation relative to the original location
            rotation_angle=original_angle_with_final_location - original_angle_with_x_axis

        if rotation_angle:
            print("Angle of rotation:", rotation_angle)
        else:
            print("Both original and final images must be captured before calculating the angle of rotation")

        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask)

        key=cv.waitKey(1)
        if key == 27:
            break

        cap.release()
        cv.destroyAllWindows()
