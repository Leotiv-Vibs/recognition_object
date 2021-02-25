# импортируем необходимые пакеты
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# -prototxt : Путь к prototxt Caffe файлу.
# -model : Путь к предварительно подготовленной модели.
# -confidence : Минимальный порог валидности (сходства) для распознавания объекта (значение по умолчанию - 20%)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# Инициализируем некоторые классы
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

color = np.random.uniform(0, 255, size=(len(classes), 3))

print("Загружаем модель")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("Начинаем стримить на камеру")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Перебирать кадры из видеопотока
while True:
    # Возьмите кадр из потоковог видеопотока
    # и измените его размер
    # иметь максимальную ширину 400 пикселей
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # возьмем размеры кадра и преобразуем его в blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Передать BLOB - объект по сети и получить
    # обнаружения и прогнозов
    net.setInput(blob)
    detections = net.forward()
    # цикл на объектами найденными
    for i in np.arange(0, detections.shape[2]):
        # Извлекать уверенность связанную с предсказанием
        confidence = detections[0, 0, i, 2]

        # Отфильтровываем слабые обнаружения для лучшея работа сети
        if confidence > args["confidence"]:
            # Находим индекс объекта и готовим рамку по размеру
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Нарисовать предсказание на рамке
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY +15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()