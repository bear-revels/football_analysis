import cv2
import matplotlib.pyplot as plt

# Load the saved frame
image_path = 'files/workbooks/extracted_frame.jpg'
image = cv2.imread(image_path)

coords = []

def onclick(event):
    print("start...")
    if event.xdata and event.ydata:
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))
        print(f"Selected coordinates: ({ix}, {iy})")
        if len(coords) >= 4:  # Assume you need at least 4 points
            plt.close()
    print('end...')

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print("Selected coordinates:", coords)
