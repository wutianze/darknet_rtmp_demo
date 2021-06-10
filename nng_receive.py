from pynng import Pair0
address = 'tcp://0.0.0.0:13131'
# in real code you should also pass recv_timeout and/or send_timeout
with Pair0(listen=address) as s0:
    while True:
        x = s0.recv()  # prints b'hi old buddy s0, great to see ya

