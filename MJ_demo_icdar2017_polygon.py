import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os
import numpy as np
import cv2
from tqdm import tqdm
from ResNeSt.resnest import resnest50
import torch.optim as optim
from labelsmoothloss import LabelSmoothingLoss
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

GroundTruth_path = os.path.join('..', 'Lizhu_TestImage/ic17_test_gt/')
img_path = os.path.join('..', 'Lizhu_TestImage/ic17_test_img/')
detect_path = os.path.join('..', 'Lizhu_TestImage/icdar2017/')
output_path = './before_ws/polygon_icdar2017_before_ws.txt'
CHECKPOINT = torch.load('./weight/ResNeSt50_english_num_2/ResNeSt50_english_num_2-1178-2.023.pth')
NUM_CLASS = 62
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 64
ITEM = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def iou(char_map , str_map, iou_num):
    total_char_size = char_map[char_map == 1].size
    in_str_char = char_map * str_map
    in_str_char_size = in_str_char[in_str_char == 1].size
    cal_iou = in_str_char_size / total_char_size
    if cal_iou >= iou_num : 
        return True
    else :
        return False

def char_and_num(character):
    if character >= 'a' and character <= 'z' :
        return True
    if character >= 'A' and character <= 'Z' :
        return True
    if character >= '0' and character <= '9' :
        return True
    
    return False

def check_candidate(candidate, character):
    if character.islower() :
        candidate.append(character.upper())
    elif character.isupper() :
        candidate.append(character.lower())
    
    if character == '0' :
        candidate.append('O')
        candidate.append('o')
    elif character == 'O' or character == 'o' :
        candidate.append('0')
    elif character == 'l' :
        candidate.append('1')  
        candidate.append('L') 
    elif character == 'l' or character == 'L' :
        candidate.append('1')
    elif character == 'i':
        candidate.append('j')
    elif character == 'j':
        candidate.append('i')
    elif character == '2' :
        candidate.append('z')  
        candidate.append('Z') 
    elif character == 'z' or character == 'Z' :
        candidate.append('2')                 
    elif character == '5' :
        candidate.append('S')  
        candidate.append('s') 
    elif character == 's' or character == 'S' :
        candidate.append('5') 
    elif character == '6' :
        candidate.append('b')
    elif character == 'b' :
        candidate.append('6')       
    elif character == '9' :
        candidate.append('q')
    elif character == 'q' :
        candidate.append('9')          
    elif not char_and_num(character):
        candidate.append('-1')
    
    return candidate

def return_candidate_char(str_list, gt_left, gt_right, pred_left, pred_right):
    candidate = []
    if len(str_list) == 0 :
        #print("ground truth is empty")
        return candidate
    elif len(str_list) == 1 :    
        candidate.append(str_list[0])
        candidate = check_candidate(candidate, str_list[0])
        return candidate
    elif len(str_list) == 2 :
        mid = gt_left + (gt_right - gt_left)/2
        if pred_left >= mid : #right char
            candidate.append(str_list[1])
            candidate = check_candidate(candidate, str_list[1])
        elif pred_right <= mid : #left char
            candidate.append(str_list[0])
            candidate = check_candidate(candidate, str_list[0])
        else :
            if pred_left - mid >= mid - pred_right :
                candidate.append(str_list[0])
                candidate = check_candidate(candidate, str_list[0])
            else :
                candidate.append(str_list[1])
                candidate = check_candidate(candidate, str_list[1])
        return candidate
    elif len(str_list) == 3 :
        mid = gt_left + (gt_right - gt_left)/2
        if pred_left >= mid : #right char
            candidate.append(str_list[1])
            candidate.append(str_list[2])
            candidate = check_candidate(candidate,str_list[1])
            candidate = check_candidate(candidate,str_list[2])
        elif pred_right <= mid : #left char
            candidate.append(str_list[0])
            candidate.append(str_list[1])
            candidate = check_candidate(candidate,str_list[0])
            candidate = check_candidate(candidate,str_list[1])
        else :
            candidate.append(str_list[0])
            candidate.append(str_list[1])
            candidate.append(str_list[2])
            candidate = check_candidate(candidate,str_list[0])
            candidate = check_candidate(candidate,str_list[1])
            candidate = check_candidate(candidate,str_list[2])
            
        return candidate
    else :
        total_len = gt_right - gt_left
        each_len_list = []
        now = gt_left
        for i in range(len(str_list)):
            each_len_list.append(now)
            now = now + total_len / len(str_list)
        
        #check included which area
        left_area_num = 0
        right_area_num = 0
        for i in range(len(each_len_list)) :
            if i+1 < len(each_len_list) and each_len_list[i] <= pred_left and each_len_list[i+1] > pred_left :
                left_area_num = i
            elif i+1 == len(each_len_list) and pred_left >= each_len_list[i] :
                left_area_num = i
            
            if i+1 < len(each_len_list) and each_len_list[i] <= pred_right and each_len_list[i+1] > pred_right :
                right_area_num = i
            elif i+1 == len(each_len_list) and pred_right >= each_len_list[i] :
                right_area_num = i   
           
           
        if left_area_num > 0 :
            left_area_num = left_area_num - 1
        if right_area_num < len(each_len_list)-1 :
            right_area_num = right_area_num + 1 
        
        for i in range(left_area_num,right_area_num+1) :
            candidate.append(str_list[i])
            candidate = check_candidate(candidate,str_list[i])
             
        #print("Return list:",candidate)        
        return candidate
        
    

# Load model and image transforms 辨識模型
model = resnest50(num_classes = NUM_CLASS, final_drop = 0, dropblock_prob = 0, pretrained = True).to(device)
model.load_state_dict(CHECKPOINT['model_state_dict'])
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY, nesterov = True)
optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
epoch = CHECKPOINT['epoch']
loss = CHECKPOINT['loss']
loss_function = LabelSmoothingLoss()
model.eval()

correct_info = 0 #detect框辨識正確 ok
incorrect_info = 0 #detect框辨識錯誤 ok
gt_total_string = 0 #gt裡總共的string數量 ok
gt_total_char = 0 #gt裡總共的char數量 ok
error_detect = 0 #detect到未在gt的框(不含###) ok
gt_not_detect_str = 0 #應該要被detect而沒有detect到框(string) ok
gt_not_detect_char = 0 #應該要被detect而沒有detect到框(char) ok
ignore_char = 0
iou_num = 0.75


for gt_filename in tqdm(os.listdir(GroundTruth_path)):
    with open(os.path.join(GroundTruth_path, gt_filename), 'r', encoding = 'utf-8') as gt_file:
        print(f'processing {gt_filename}...')
        
        # load detect information list
        detect_list = []
        detect_filename = gt_filename.replace("gt", "res")
        root_path = os.path.join(detect_path + detect_filename)
        # print(detect_path)
        # print(detect_filename)
        # print(root_path)
        # print(os.getcwd())
        if os.path.isfile(root_path): 
            detect_file = open(root_path)
            for line in detect_file :
                line=line.strip('\n')
                line_list = line.split(',')[:-2]
                line_list = list(map(int, line_list))
                line_list = np.array(line_list).reshape(-1,2)
                detect_list.append(line_list)
                
            detect_file.close
        else:
            print(detect_filename," not find!")
            
            continue
            
        #load image
        img_filename = gt_filename.replace("gt_", "");
        img_filename = img_filename.replace(".txt", ".jpg");
        #img = Image.open(img_path+img_filename).convert('RGB')
        img = cv2.imread(img_path + img_filename)

        #print("pred bbox num:",len(detect_list))
        
        #now detect_list is detect information
        for index in gt_file:
       
            index=index.strip('\n')
            index_list = index.split(',')
            
            bbox = index_list[:8]
            bbox = list(map(int, bbox))
            bbox = np.array(bbox).reshape(-1,2)
            if len(index_list) > 10 :
                for i in range(10,len(index_list)) :
                    index_list[9] = index_list[9] + "," + index_list[i]
            
            #bbox = [ min(int(index_list[0]),int(index_list[6])),min(int(index_list[1]),int(index_list[3])), max(int(index_list[2]),int(index_list[4])), max(int(index_list[7]),int(index_list[5]))]
            #處理gt 的bbox 畫在圖上 方便做iou
            str_map = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(str_map,[bbox],(1,1,1))
                
            if index_list[8] == "Latin" and index_list[9] != "###" : #使用完要刪除
                gt_total_string = gt_total_string + 1

                character_num = 0 
                for character in index_list[-1] :
                    if char_and_num(character) :
                        character_num = character_num + 1
                
                gt_total_char = gt_total_char + character_num
                
                #check 方向
                x_len = int(index_list[2])-int(index_list[0])
                y_len = int(index_list[3])-int(index_list[1])
                if abs(x_len) >= abs(y_len) :
                    is_horizontal = True 
                    
                    if x_len < 0 :
                        gt_str = index_list[9][::-1]
                    else :
                        gt_str = index_list[9]
                else:
                    is_horizontal = False 
                    if y_len < 0 :
                        gt_str = index_list[9][::-1]
                    else :
                        gt_str = index_list[9]  
                    
                find_num = 0 
                correct_num = 0
                new_detect_list = []
                for index_detect in detect_list[:]:
                    #print("===================",len(detect_list),"===================")
                    char_map = np.zeros(img.shape, dtype=np.uint8)
                    cv2.fillPoly(char_map,[index_detect],(1,1,1))
                    
                    if iou( char_map , str_map , iou_num ):
                        
                        mask = img * char_map
                        
                        crop_img = mask[min(index_detect[:,1]):max(index_detect[:,1]),min(index_detect[:,0]):max(index_detect[:,0])]
                        try :
                            crop_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                        except:
                            continue
                        
                        
                        crop_img = F.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = F.InterpolationMode.BICUBIC)
                        crop_img = np.array(crop_img)
                        crop_img = torch.from_numpy(crop_img).unsqueeze(0)
                        crop_img_gray = torch.mean(crop_img.float(), dim = 3, keepdim = True)
                        crop_img_gray = crop_img_gray.permute(0, 3, 1, 2)
                        crop_img_gray = crop_img_gray.to(device)
                        outputs = model(crop_img_gray)
                        outputs.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

                        # Greedy decoding
                        pred = outputs.softmax(-1)[0]
                        #print('==========pred==========')
                        #print(pred)
                        #print('==========pred.shape==========')
                        #print(pred.shape)
                        #label, confidence = torch.max(pred.data, 1)
                        print('==========torch.max(pred.data, 1)==========')
                        print(torch.max(pred.data, 0)[1])
                        label_index = torch.max(pred.data, 0)[1]
                        label = ITEM[label_index.item()]
                        print('label:', label)
                        #label, confidence = model.tokenizer.decode(pred)
                        print('Decoded label = {}'.format(label[0]))

                        if len(label[0]) == 0 :
                            incorrect_info = incorrect_info + 1
                        else :
                            #橫向
                            if is_horizontal :
                                candidate_list = return_candidate_char(gt_str, min(int(index_list[0]), int(index_list[6])), max(int(index_list[2]),int(index_list[4])),min(index_detect[:,0]),max(index_detect[:,0]))
                            else :#直向
                                candidate_list = return_candidate_char(gt_str,min(int(index_list[1]),int(index_list[3])),max(int(index_list[7]),int(index_list[5])),min(index_detect[:,1]),max(index_detect[:,1]))
                    
                            for each_label in label[0] :
                                find_num = find_num + 1
                                
                                if not char_and_num(each_label) and '-1' in candidate_list : #辨識結果非英數字 且候選裡也存在符號
                                    correct_info = correct_info + 1
                                elif not char_and_num(each_label) and '-1' not in candidate_list : #辨識結果非英數字 但候選裡沒有符號
                                    incorrect_info = incorrect_info + 1
                                elif each_label in candidate_list :
                                    correct_info = correct_info + 1
                                    correct_num = correct_num + 1
                                    candidate_list.remove(each_label)
                                elif '-1' in candidate_list :
                                    ignore_char = ignore_char + 1
                                else :
                                    incorrect_info = incorrect_info + 1
                        
  
                    else :
                        new_detect_list.append(index_detect)

                detect_list = new_detect_list.copy()

                
                if find_num == 0 :
                    gt_not_detect_str = gt_not_detect_str + 1
                else :
                    find_num = character_num - find_num
                    if find_num > 0 :
                        gt_not_detect_char = gt_not_detect_char + find_num
                        
                    if correct_num > character_num :
                        incorrect_info = incorrect_info + correct_num - character_num
                        correct_info = correct_info - (correct_num - character_num)                    
                
            else : ### Don't care. Remove included detect bbox
                new_detect_list = []
                for index_detect in detect_list[:]:
                    char_map = np.zeros(img.shape, dtype=np.uint8)
                    cv2.fillPoly(char_map,[index_detect],(1,1,1))
                    
                    if not iou( char_map , str_map , iou_num ):
                        new_detect_list.append(index_detect)
                        #detect_list.remove(index_detect)
                        #print("don't care")
                    #else :
                    #    new_detect_list.append(index_detect)
                
                detect_list = new_detect_list.copy()
                
        error_detect = error_detect + len(detect_list)
        #print("detect_list size:",len(detect_list))
       
    
print("done!")

accuracy = (correct_info / gt_total_char) * 100
model_accuracy = (correct_info / (correct_info + incorrect_info)) * 100
detect_error_rate = (error_detect / (correct_info + incorrect_info + error_detect)) * 100
not_detect_str_rate = (gt_not_detect_str / gt_total_string) * 100
not_detect_char_rate = (gt_not_detect_char / gt_total_char)*100

#score output

f = open(output_path, 'w')

f.write("correct_info:")
f.write(str(correct_info))
f.write("\nincorrect_info:")
f.write(str(incorrect_info))
f.write("\ngt_total_string:")
f.write(str(gt_total_string))
f.write("\ngt_total_char:")
f.write(str(gt_total_char))
f.write("\nerror_detect:")
f.write(str(error_detect))
f.write("\ngt_not_detect_str:")
f.write(str(gt_not_detect_str))
f.write("\ngt_not_detect_char:")
f.write(str(gt_not_detect_char))
f.write("\nignore_char:")
f.write(str(ignore_char))
f.write("\n= = = = = = = = = = = = = = = = = =\n")

f.write("\nAccuracy:")
f.write(str(accuracy))
f.write("\nModel accuracy:")
f.write(str(model_accuracy))
f.write("\nDetect error rate:")
f.write(str(detect_error_rate))
f.write("\nNot detect string rate:")
f.write(str(not_detect_str_rate))
f.write("\nNot detect char rate:")
f.write(str(not_detect_char_rate))
f.close()

print("txt write done!")

