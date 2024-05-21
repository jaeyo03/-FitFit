from gradio_client import Client, file
import os
client = Client("rlawjdghek/StableVITON")

image_folder = "./baby_image" 
cloth_folder = "./cloth"

for filename in os.listdir(image_folder):
	if filename.endswith(".jpg"):
		image_path = os.path.join(image_folder, filename)
		result = client.predict(
				vton_img=file(image_path),
				garm_img=file("/Users/junghyunkim/Desktop/capstone/cloth/00111_00.jpg"),
				n_steps=20,
				is_custom=False,
				api_name="/process_hd"
		)
	print(result)