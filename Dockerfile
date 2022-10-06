FROM python:3.10.7-bullseye

WORKDIR /app

RUN apt update && apt install -y cmake ffmpeg libsm6 libxext6

COPY . .

RUN pip3 install -r requirements.txt

# CMD [ "python3", "main_video.py", "--src_img", "imgs/test1.jpg", "--video_path", "videos/jwick.mp4", "--show", "--correct_color", "--warp_2d", "--save_path", "output/vid1.avi"]
CMD [ "pwd"]