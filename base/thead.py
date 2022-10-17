import threading, cv2
from .funsion_base import *
from .models import *
from PIL import Image
import base64
import io
from tensorflow.keras.utils import array_to_img

class CapChaTikTokTron(threading.Thread):
    def __init__(self, img_small, img_big, room, user, name_img):
        self.img_small = img_small
        self.img_big = img_big
        self.room = room
        self.user = user
        self.name_img = name_img
        threading.Thread.__init__(self)

    def run(self):
        code_pre = None
        img_byte = io.BytesIO()
        try:
            code_small = base64.b64encode(self.img_small)
            code_big = base64.b64encode(self.img_big)
            
            code_small = code_small.decode('utf-8')
            code_big = code_big.decode('utf-8')
            

        
            nho = base64.b64decode(code_small)
            img_small = Image.open(io.BytesIO(nho))


            lon = base64.b64decode(code_big)
            img_big = Image.open(io.BytesIO(lon))


            small = img_small.convert('RGB')
            big = img_big.convert('RGB')

            small = small.resize((211,211))
            big = big.resize((347,347))

            small_img = np.array(small, dtype= np.uint8)
            big_img = np.array(big, dtype= np.uint8)

            small_img=array_to_img(small_img)
            big_img=array_to_img(big_img)
            
            
            img = capChaTronTikTok(small_img, big_img)
            

            img.save(img_byte, format='PNG')
            img_byte = img_byte.getvalue()
            
            code = base64.b64encode(img_byte).decode('utf-8')
            # lưu vào database 
            CapChaTikTok.objects.create(
                user = self.user,
                room = self.room,
                name = self.name_img,
                codeImg_small = code_small,
                codeImg_big = code_big,
                codeImg_pre = code,
            )
            img_capCha = CapChaTikTok.objects.get(name = self.name_img)
            code_pre = img_capCha.codeImg_pre
            print("predict capchatitok tron <thead>: ",type(code_pre))

        except Exception as e:
            print("error capchatitok tron.")
            print(e)
        return code_pre