import cv2

trained_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video=cv2.VideoCapture("3249935-uhd_3840_2160_25fps.mp4")

while True:
    Success,Frame=video.read()

    if Success==True:
        face=trained_data.detectMultiScale(Frame)
        for x,y,w,h in face:
            cv2.rectangle(Frame, (x, y), (x + w, y + h), (100, 0, 0), 2)
            # print(img)

            cv2.imshow("out", Frame)

            cv2.waitKey(1)
    else:
        break