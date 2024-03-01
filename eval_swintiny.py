import os
import torch
import torch.utils.data.distributed
import matmodel
from einops import rearrange
import cv2
import numpy as np

img_path='./image/'
alpha_path='./alpha/'
os.makedirs(alpha_path,exist_ok=True)

if __name__ == '__main__':
    resol=640
    maxp=20
    e2ehim = matmodel.E2EHIMST()
    e2ehim.load_state_dict(torch.load("./swintiny.ckpt",map_location='cpu'),strict=False)
    e2ehim = e2ehim.cuda()
    e2ehim.eval()
    for file in os.listdir(img_path):
        img=cv2.imread(img_path+file)
        h,w,c=img.shape
        imgp=cv2.resize(img,(640,640),interpolation=cv2.INTER_LANCZOS4)
        imgp=np.array(imgp,np.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = imgp[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        image = torch.from_numpy(image).sub_(mean).div_(std)[None,:,:,:]
        image=image.cuda()
        with torch.no_grad():
            x4o,x1all,maskt= e2ehim(image)
        maskt=maskt[0]
        maskt=rearrange(maskt,'(t  c2) h w-> t c2 h w',c2=3)
        maskt=torch.softmax(maskt,1)
        maskt=torch.argmax(maskt,1)
        maskt=maskt.cpu().numpy()
        x1m=x1all.cpu().numpy()
        x1m=np.clip(x1m,0,1)
        xp=x1m[0]
        pidx=0
        for x in range(maxp):
            psel=x4o[0,x]
            if psel[1]>psel[0]:
                pmat=xp[x]
                ptri=maskt[x]
                pmat=np.clip(pmat,0,1)
                pmat=pmat*255.*(ptri==1)+(ptri==2)*255.
                pmat=np.array(pmat,np.uint8)
                pmat=cv2.resize(pmat,(w,h),interpolation= cv2.INTER_LANCZOS4)
                if (np.sum(pmat)/255.)>1000.:
                    cv2.imwrite(alpha_path+file+"_{:0>2d}.png".format(pidx),pmat)
                    pidx+=1

