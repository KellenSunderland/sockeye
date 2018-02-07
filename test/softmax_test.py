import mxnet as mx

x = mx.nd.random.uniform(0, 1111, (34000, 1), ctx=mx.gpu(0))
ops = []

for i in range(0, 5000):
    y = mx.nd.softmax(x)
    ops.append(y)

[op.wait_to_read() for op in ops]

