import keras.backend as b
import keras

def huber_loss(x, y):
    e = x - y
    loss = b.mean(b.sqrt(1 + b.square(e)) - 1, axis=-1)
    return loss


def huber_loss_print(x, y):
    x = b.print_tensor(x, message="x= ")
    y = b.print_tensor(y, message="y= ")
    e = x - y
    e = b.print_tensor(e, message="e= ")
    loss = b.mean(b.sqrt(1 + b.square(e)) - 1, axis=-1)
    loss = b.print_tensor(loss, message="loss= ")
    return loss

keras.losses.huber_loss = huber_loss
keras.losses.huber_loss_print = huber_loss_print