from tesserocr import PyTessBaseAPI
import os
from PIL import Image  

file_dir = r'D:\卢艳\父亲节微信主体截图'

images = ['1.jpg', '2.jpg', '3.jpg']

with PyTessBaseAPI(lang='chi_sim') as api:
    for img in images:
        img=os.path.join(file_dir,img)
        api.SetImageFile(img)
        text=api.GetUTF8Text()
        print(text)

        t_list = text.split('\n')
        print(t_list)


from tesserocr import get_languages

print(get_languages(r'D:\Tesseract-OCR\tessdata'))  # or any other path that applies to your system