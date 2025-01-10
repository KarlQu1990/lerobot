import cv2
import sys
import time
import logging
import multiprocessing as mp
from multiprocessing import Queue, Event
from multiprocessing.synchronize import Event as MpEvent



def decode(index, queue: Queue):
    cap = cv2.VideoCapture(f"/dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 25.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"video {index}, is opened: {cap.isOpened()}")

    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret:
            print("get frame failed, index=", index)
            break
        
        cv2.imshow(f"frame{index}", frame)
        if cv2.waitKey(1) == 27:
            break
        
        queue.put((time.time(), frame))

    print("read frame ended:", index)
    cap.release()


if __name__ == "__main__":
    indexes = list(map(int, sys.argv[1].split(",")))

    queues = []
    # stop_event = Event()
    for index in indexes:
        queue = Queue()
        proc = mp.Process(target=decode, args=(index,queue))
        proc.start()
        queues.append(queue)

    try:
        while True:
            for index, queue in zip(indexes, queues):
                try:
                    if not queue.empty():
                        t, frame = queue.get()
                        print(f"get frame {index}, t: {t}, shape: {frame.shape}")
                except Exception:
                    pass
                
            time.sleep(0.01)
    except Exception as e:
        time.sleep(1)
        sys.exit(0)
    