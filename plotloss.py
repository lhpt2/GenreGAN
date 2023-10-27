import matplotlib.pyplot as plt
import pandas as pd
import sys

# 0 - epoch, 1 - dloss, 2 - dreward, 3 - gloss, 4 - idloss, 5 - learning rate
# 0 - epoch, 1 - dloss_fake, 2 - dloss_real, 3 - gloss, 4 - idloss, 5 - learning rate

# 0 epoch, 1 - dloss, 2- drloss, 3 - dfloss, 4 - gloss, 5 - sloss, 6 - idloss, 7 - vloss, 8 - lr

csvdata = pd.read_csv(sys.argv[1], header=0)

num = csvdata[0][1:-1]
dloss = csvdata[1][1:-1]
gloss = csvdata[3][1:-1]
#gloss = csvdata[4][1:-1]
vloss = csvdata[7][1:-1]

print(num)
#print(dloss)
#print(gloss)


# fig, ax = plt.subplots(figsize=(100, 50))
# ax.plot(dloss, label="Discriminator Loss")
# ax.plot(gloss, label="Generator Loss")
# plt.legend()
# plt.show()