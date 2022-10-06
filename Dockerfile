FROM python:3.10.7-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt update && apt install -y cmake ffmpeg libsm6 libxext6

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "main_video.py", "--src_img", "imgs/test1.jpg", "--video_path", "imgs/jwick.mp4", "--show", "--correct_color", "--warp_2d", "--save_path", "video/vid1.avi"]