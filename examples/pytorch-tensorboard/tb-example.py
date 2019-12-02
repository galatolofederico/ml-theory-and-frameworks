from tensorboardX import SummaryWriter
import numpy as np
import random

writer = SummaryWriter()

for i in range(1000):
    writer.add_scalar("something", (random.random()-0.5) + np.sin(4*np.pi*i/1000), i)
