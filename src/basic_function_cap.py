import cv2
import matplotlib.pyplot as plt

def show_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16,6))
        plt.imshow(frame)
        plt.show()
    else:
        print('Frame not found')

def get_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame
    else:
        return ret, None

def crop_frame(frame, x, y, w, h, show=False):
    cropped = frame[x:x+w, y:y+h, :]
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(cropped)
        plt.show()
    return cropped

def crop_and_save_video(cap, x, y, w, h, output_file, begin_frame=False, end_frame=False):
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    if not output_file.endswith('.mp4'):
        output_file += '.mp4'
    out = cv2.VideoWriter(output_file, # or .mkv
        fourcc, cap.get(cv2.CAP_PROP_FPS),
        (h, w))
    if not begin_frame:
        begin_frame = 0
    if not end_frame:
        end_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(begin_frame, end_frame):
        ret, frame = get_frame(cap, i)
        if ret:
            frame = crop_frame(frame, x, y, w, h)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        else:
            break
    out.release()
    cv2.destroyAllWindows()