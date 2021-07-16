ps -a|grep n|grep -v client|awk '{print $1}'|xargs kill -9
sleep 3
(uvicorn darknet_websocket_demo:app --host 0.0.0.0 --port 1935 --reload) &
sleep 3
curl "http://10.118.0.45:1935/start"
