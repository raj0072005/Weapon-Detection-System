import cv2

for backend_name, backend in [('CAP_ANY', cv2.CAP_ANY), ('CAP_DSHOW', cv2.CAP_DSHOW), ('CAP_MSMF', cv2.CAP_MSMF)]:
    for index in [0, 1, 2, -1]:
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        print(f'{backend_name} index={index} opened={opened}')
        if opened:
            ok, frame = cap.read()
            print(f'  read={ok} shape={None if frame is None else frame.shape}')
        cap.release()
