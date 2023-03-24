import os
import random
import shutil

# Diretório das imagens
# dir_path = "D:\Dataset\BID_Dataset\CNH_Frente"
# dir_path = "D:\Dataset\BID_Dataset\CNH_Verso"
# dir_path = "D:\Dataset\BID_Dataset\CPF_Frente"
dir_path = "D:\Dataset\BID_Dataset\CPF_Verso"

# Percentual de imagens que serão usadas para validação
val_percent = 20

# Diretório onde as imagens de treinamento serão salvas
train_path = os.path.join(dir_path, "train")
os.makedirs(train_path, exist_ok=True)

# Diretório onde as imagens de validação serão salvas
val_path = os.path.join(dir_path, "val")
os.makedirs(val_path, exist_ok=True)

# Lista com o nome de todos os arquivos de imagem na pasta
img_files = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]

print(dir_path)
print(len(img_files))

# Embaralha a lista de arquivos
random.shuffle(img_files)

# Calcula o número de imagens para validação
num_val = int(len(img_files) * val_percent / 100)


# Copia as imagens para a pasta de treinamento
for file_name in img_files[num_val:]:
    src_path = os.path.join(dir_path, file_name)
    dst_path = os.path.join(train_path, file_name)
    shutil.copyfile(src_path, dst_path)

# Copia as imagens para a pasta de validação
for file_name in img_files[:num_val]:
    src_path = os.path.join(dir_path, file_name)
    dst_path = os.path.join(val_path, file_name)
    shutil.copyfile(src_path, dst_path)
