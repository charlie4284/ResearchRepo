docker run -p 8888:8888 --gpus all -it -v `pwd`:`pwd` -v /dev/video0:/dev/video0 -w `pwd` ml
sudo docker run -p 8888:8888 --rm -ti --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" --gpus all -it -v `pwd`:`pwd` -v /dev/video0:/dev/video0 -w `pwd` ml
