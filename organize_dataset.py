import os

folder_path = "D:\Dataset\BID_Dataset"

for folder in os.listdir(folder_path):

    file_name_folder = os.path.join(folder_path, folder)

    for file_name in os.listdir(file_name_folder):
        # if i == 0:
        #    print(file_name)
        if file_name.endswith("_gt_segmentation.jpg" ) or file_name.endswith("_gt_ocr.txt"): # verifica se o nome do arquivo termina com "_gt_segmentation"
            file_path = os.path.join(file_name_folder, file_name) # obtém o caminho completo do arquivo
            os.remove(file_path) # exclui o arquivo