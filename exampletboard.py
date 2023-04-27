from torch.utils.tensorboard import SummaryWriter
import time 


if __name__ == "__main__":
    writer = SummaryWriter()
    x = range(100)
    for i in x:
        writer.add_scalar('y=x*2', i ** 2, i)
        writer.add_scalar('y=x*3', i ** 3, i)
        time.sleep(3)
    writer.close()

