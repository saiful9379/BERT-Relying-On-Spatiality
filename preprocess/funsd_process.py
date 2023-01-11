
import os
import json
import glob
import cv2
import random
import shutil

def draw_key_value_other(
    image, funsd_data_key_value, funsd_data_other, file_name):
    
    lenght_of_pair = int(len(funsd_data_key_value)/2) 
    # print(len(funsd_data_key_value), lenght_of_pair)
    thickness = 2
    start, end = 0, 1
    for _ in range(lenght_of_pair):
        color = [random.randint(0, 255) for _ in range(3)]
        # print(start, end)
        key = funsd_data_key_value[start]
        value = funsd_data_key_value[end]

        # print(key)
        key_bbox = key["box"]
        value_bbox = value["box"]
        words = key["words"]+value["words"]
        kx1, ky1, kx2, ky2 = key_bbox
        vx1, vy1, vx2, vy2 = value_bbox
        s_point = int((ky2-ky1)/2)
        e_point = int((vy2-vy1)/2)

        cv2.arrowedLine(image, (kx2, ky1+s_point), (vx1, vy1+e_point), color, thickness)
        cv2.rectangle(image, (kx1,ky1), (kx2, ky2), color, thickness)
        cv2.rectangle(image, (vx1,vy1), (vx2, vy2), color, thickness)
        for word in words:
            wx1, wy1, wx2, wy2 = word["box"]
            cv2.rectangle(image, (wx1,wy1), (wx2, wy2), color, 1)

        start = end+1
        end = start+1
    for other in funsd_data_other:
        # print(other)
        ox1, oy1, ox2, oy2 = other["box"]
        # print(other)
        o_words = other["words"]
        # print(o_words)
        for oword in o_words:
            owx1, owy1, owx2, owy2 = oword["box"]
            cv2.rectangle(image, (owx1,owy1), (owx2, owy2), (0,255,255), 1)
        cv2.rectangle(image, (ox1,oy1), (ox2, oy2), (0,0,255), 1)
    
    cv2.imwrite(os.path.join("logs", file_name), image)

def read_json(json_path:str='')->dict:
    with open(json_path) as json_file:
        data = json.load(json_file)
    print(len(data))
    return data

def write_json(dataset_dict, json_file_path:str=""):
    with open(json_file_path, 'w') as outfile:
        json.dump(dataset_dict, outfile, ensure_ascii=False, indent=4)

def get_data_format(key_data, key_word_data, key, idx):
    data_list = []
    for key_d in key_data:
        # print(key)
        # print(key_d["shape_attributes"])
        shape_attributes = key_d["shape_attributes"]
        region_attributes = key_d["region_attributes"]
        x1, y1, x2, y2 = shape_attributes["x"], shape_attributes["y"], \
            shape_attributes["x"]+shape_attributes["width"], shape_attributes["y"]+shape_attributes["height"]
        text = region_attributes["text"]
        data = {
                "text": text,
                "box": [x1, y1, x2, y2],
                "linking": [],
                "label": key,
                "words": []
            }
        # words = []
        for word in key_word_data:
            w_shape_attributes = word["shape_attributes"]
            w_region_attributes = word["region_attributes"]
            word_text = w_region_attributes["text"]
            wx1, wy1, wx2, wy2 = w_shape_attributes["x"], w_shape_attributes["y"], \
            w_shape_attributes["x"]+w_shape_attributes["width"], w_shape_attributes["y"]+w_shape_attributes["height"]
            # print("==========", word)
            word_dict = {
                "text": word_text,
                "box": [wx1, wy1, wx2, wy2]
            }
            data["words"].append(word_dict)
        data_list.append(data)
        idx += 1
    return data_list, idx



def key_and_value_mapping(key_dict, key_word_dict, value_dict, value_word_dict):
    print(len(key_dict), len(key_word_dict), len(value_dict), len(value_word_dict))
    idx = 0
    data_list, last_index = [], 0

    for key, key_data in key_dict.items():
        # key operation
        key_word_data = key_word_dict[key]
        key_data_list, idx = get_data_format(key_data, key_word_data, "key", idx)
        key_data_list[0]["id"] = idx
        linking = [idx]
        # value operation
        value_data = value_dict[key]
        value_word_data = value_word_dict[key]
        value_data_list, idx = get_data_format(value_data, value_word_data, "value", idx)
        value_data_list[0]["id"] = idx
        linking.append(idx)
        # key and value linking
        key_data_list[0]["linking"].append(linking)
        value_data_list[0]["linking"].append(linking)

        data_list.extend(key_data_list)
        data_list.extend(value_data_list)

        idx += 1
        last_index=idx
    return data_list, last_index

def other_and_other_word_mapping(other_dict, other_word_dict, last_index):
    other_data_list = []
    for key, other in other_dict.items():
        other_word_data = other_word_dict[key]
        value_data_list, _ = get_data_format(
            other, other_word_data, "other", last_index) 
        last_index+=1
        value_data_list[0]["id"] = last_index
        other_data_list.append(value_data_list[0])
    return other_data_list
    
def vgg_to_funsd_data_conversion(
    data_dict, 
    data_type, 
    output_dir,
    input_img_path
    ):
    print(output_dir)
    print(data_type)
    # exit()
    output_json_dir  = os.path.join(output_dir, data_type, "annotations")
    # img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(output_json_dir,  exist_ok= True)
    # exit()
    # ndata_dict = {}
    for key, value in data_dict.items():
        file_name = value["filename"]
        print(file_name)
        img = cv2.imread(os.path.join(input_img_path, file_name))
        regions = value["regions"]
        funsd_formate = {"form":[]}
        key_dict, key_word_dict, value_dict, value_word_dict = {}, {}, {}, {}
        other_dict, other_word_dict = {}, {}
        for region in regions:
            key = region["region_attributes"]["layout"]
            _id = region["region_attributes"]["id"]
            key_id = f"{key}_{_id}"

            if key == "Key":
                if _id not in key_dict:
                    key_dict[_id] =  [region]
                else:
                    key_dict[_id].append(region)
            if key=="key_words":
                if _id not in key_word_dict:
                    key_word_dict[_id] =  [region]
                else:
                    key_word_dict[_id].append(region)
            if key=="Value":
                if _id not in value_dict:
                    value_dict[_id] =  [region]
                else:
                    value_dict[_id].append(region)
            if key=="value_words":
                if _id not in value_word_dict:
                    value_word_dict[_id] =  [region]
                else:
                    value_word_dict[_id].append(region)
            if key=="Other":
                if _id not in other_dict:
                    other_dict[_id] =  [region]
                else:
                    other_dict[_id].append(region)

            if key=="other_words":
                if _id not in other_word_dict:
                    other_word_dict[_id] =  [region]
                else:
                    other_word_dict[_id].append(region)

        funsd_data_key_value, last_index = key_and_value_mapping(key_dict, key_word_dict, value_dict, value_word_dict)
        funsd_data_other = other_and_other_word_mapping(other_dict, other_word_dict, last_index)
        draw_key_value_other(
            img, funsd_data_key_value, funsd_data_other, file_name
            )
        # print(funsd_data_other)
        total_data = funsd_data_key_value + funsd_data_other
        for t_data in total_data:
            funsd_formate["form"].append(t_data)
        output_json_path = os.path.join(output_json_dir, file_name[:-4]+".json")
        write_json(funsd_formate, output_json_path)


if __name__ == "__main__":
    input_path = "/media/nsl8/hdd/saiful/layoutlm/raw_code/data/libor_editable_cluster_chunks"
    data_type = ["testing_data", "training_data"]
    for dt in data_type:
        print("Data Conversion : ", dt)
        json_file = os.path.join(input_path, dt, "data_filtered.json")
        img_path = os.path.join(input_path, dt, "images")
        # image_list = [os.path.basename(i) for i in glob.glob(img_path+"/*")]
        data = read_json(json_file)
        vgg_to_funsd_data_conversion(data, dt, input_path, img_path)
        print("Done")
        # break
