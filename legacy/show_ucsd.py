from PIL import Image
import matplotlib.pyplot as plt

image = Image.open(
    "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001/050.tif"
)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
