# DySUN
UTAR Final Year Project 2

Download server.py, yeelight_server.py and the tensorflow model file.

:On Server site.
create folder with following structure:
/server

  ~/model_param
    ~/model.h5
    ~/model.json
    
  ~/static
  
    ~/icon.png
    ~/styles.css
  ./template
    ./config.html
  ./uploads
  ./docker-compose.yml
  ./Dockerfile
  ./requirements.txt
  ./server.py

:save yeelight_server.py on the microcontroller
  :remember to change the path to your tensorflow model in server.py

:use docker-compose build to create the image and start container
  :remember to change the ip of your server in yeelight_server.py

:start server first, then run yeelight_server.py


