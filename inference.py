from PIL import Image
from dataset.main import TrocrPredictor

# load images
image_names = ["data/img1.png", "data/img2.png"]
images = [Image.open(img_name) for img_name in image_names]

# directly predict on Pillow Images or on file names
model = TrocrPredictor()
predictions = model.predict_images(images)
predictions = model.predict_for_file_names(image_names)

# print results
for i, file_name in enumerate(image_names):
    print(f'Prediction for {file_name}: {predictions[i]}')