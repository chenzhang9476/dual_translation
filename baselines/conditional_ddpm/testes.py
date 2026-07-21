import json
import os
#{"pCT": item_A, "CBCT": item_B}

task_id=2

if task_id==1:
    task_target="mr"
elif task_id==2:
    task_target="cbct"

root=f"../synthRAD2025_Task{task_id}_Train_process/"
pth_json=root+"split.json"

# JSONファイルを開いて読み込む
with open(pth_json, 'r') as file:
    name_img = json.load(file)["train_cases"]

root_img=root+"HN/"
# 読み込んだデータを表示

files_A = []
files_B = []

for i in name_img:
    dir_img=root_img+f"{i}/"
    num_img=(len(os.listdir(dir_img))//3)
    print(num_img)
    name_A=[dir_img+f"ct_{str(j).zfill(3)}.npy" for j in range(num_img)]
    name_B=[dir_img+f"{task_target}_{str(j).zfill(3)}.npy" for j in range(num_img)]
    
    files_A+=name_A
    files_B+=name_B

files_A=sorted(files_A)
files_B=sorted(files_B)

print(files_B)