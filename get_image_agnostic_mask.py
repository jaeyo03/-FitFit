# import os
# from PIL import Image
# import numpy as np

# def process_images(input_folder, output_folder):
#     # Ensure output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Process each image in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(input_folder, filename)
#             image = Image.open(image_path)
#             img_array = np.array(image)

#             # Apply the mask for the specific gray color [128, 128, 128]
#             mask = np.all(np.logical_and(img_array >= [120, 120, 120], img_array <= [136, 136, 136]), axis=-1)
#             new_img_array = np.zeros_like(img_array)
#             new_img_array[mask] = [255, 255, 255]  # Set exact gray areas to white
#             new_img_array[~mask] = [0, 0, 0]       # Set all other areas to black

#             # Convert back to an image
#             new_image = Image.fromarray(new_img_array)
            
#             # Save the processed image in the output folder
#             new_filename = filename.split(".")[0] + "_mask.png"
#             output_path = os.path.join(output_folder, new_filename)
#             new_image.save(output_path)
#             print(f"Processed and saved: {output_path}")

# # Example usage:
# process_images('/Users/junghyunkim/HR-VITON/test/myagnostic', '/Users/junghyunkim/HR-VITON/test/myagnostic_mask')


import json
from os import path as osp
import os
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    # r = 20
    agnostic = img.copy()
    agnostic = Image.new('RGB', (768, 1024), 'black')
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a/16)+1

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'white', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'white', 'white')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'white', 'white')
    black_img = Image.new('RGB', (768, 1024), 'black')
    agnostic.paste(black_img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(black_img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

if __name__=="__main__":
    data_path = './test'
    output_path = './test/myagnostic_mask'
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        print(im_name)
        if im_name.endswith('.jpg'):
            # load pose image
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            
            try:
                with open(osp.join(data_path, 'openpose_json', pose_name),'r',encoding='utf-8') as f:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1, 3))[:, :2]
            except IndexError:
                print(pose_name)
                continue

            # load agnostic image
            im = Image.open(osp.join(data_path,'image',im_name))
            label_name = im_name.replace('.jpg', '.png')
            im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

            agnostic = get_img_agnostic(im,im_label,pose_data)
            im_name = im_name.split(".")[0] + "_mask.png"
            agnostic.save(osp.join(output_path, im_name))