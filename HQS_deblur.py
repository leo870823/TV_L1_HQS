import numpy as np
import cv2 
import math

# ==== image blurring ====
def psf2otf(psf, outSize):
    #Ref: https://blog.csdn.net/weixin_43890288/article/details/105676416
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

def convolution(image,kernel):
    h,w=image.shape
    #F_B=np.fft.fft2(kernel,(h,w) )
    F_B=psf2otf(kernel,(h,w) )
    F_IMG=np.fft.fft2(image)
    restored = np.fft.ifft2(F_B*F_IMG).real
    return restored

def color_convolution(image,kernel):
    color=np.zeros(image.shape)
    for i in range(0,3):
        color[:,:,i]=convolution(image[:,:,i],kernel)
    return color

# ==== image deblurring ====
def ProxF(_y,_lambda):
    normalized=np.maximum(np.abs(_y[:,:,0])+np.abs(_y[:,:,1]), _lambda)
    #normalized=np.maximum(np.sqrt(_y[:,:,0]*_y[:,:,0]+_y[:,:,1]*_y[:,:,1]),255.0)
    return _y/np.stack([normalized,normalized],axis=2)
def uniform_deblur_HQS(image,kernel,Epoch):
    cv2.imwrite("result/out/original.png",image) 
    #parameter 
    _lambda=0.01
    h,w=image.shape
    #variable
    x=image
    #FFT term
    F_I=np.fft.fft2(image,(h,w))  
    F_B=psf2otf(kernel,(h,w))          #Blur kernel
    F_Kx=psf2otf(np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),(h,w))#Gradient matrix
    F_Ky=psf2otf(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),(h,w))#Gradient matrix
    #F_B=psf2otf(kernel,(h,w) )
    Denom=np.abs(F_B)*np.abs(F_B)+_lambda*(np.abs(F_Kx)*np.abs(F_Kx)+np.abs(F_Ky)*np.abs(F_Ky))
    for k in range(0,Epoch):#HQS implementation
        fx = cv2.filter2D(x, dst=-1,ddepth=-1, kernel=np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), anchor=(-1, -1))
        fy = cv2.filter2D(x, dst=-1,ddepth=-1, kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), anchor=(-1, -1))
        dx=np.stack([fx,fy],axis=2)
        z=dx-ProxF(dx,_lambda)
        F_Z=np.fft.fft2(z)
        K_TZ=np.matrix.conj(F_Kx)*F_Z[:,:,0]+np.matrix.conj(F_Ky)*F_Z[:,:,1]
        Nom1=np.matrix.conj(F_B)*F_I+K_TZ*_lambda
        x= Nom1/Denom 
        x=np.fft.ifft2(x).real
        cv2.imwrite("result/out/blurred_%d.png"%(k),x) 
    return x
# ==== Evaluation metric ====
def compute_PSNR(img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    

if __name__ == '__main__':
    #read image
    kernel=cv2.imread("kernels/fading.png", cv2.IMREAD_GRAYSCALE)
    kernel=kernel/np.sum(kernel)
    img=cv2.imread("images/church.jpg")#, cv2.IMREAD_GRAYSCALE
    #scale
    scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Blurred image
    blurred_image=color_convolution(image,kernel) 
    cv2.imwrite("result/out/blurred_color_image.png",blurred_image)
    #TV-L2 Deblur
    print("Before blurred:",compute_PSNR(blurred_image,image)) 
    for i in range(0,3):
        #blurred_image[:,:,i]=uniform_deblur_PD(blurred_image[:,:,i],kernel,20)
        blurred_image[:,:,i]=uniform_deblur_HQS(blurred_image[:,:,i],kernel,20)
    cv2.imwrite("result/out/deblurred_color_image.png",blurred_image)
    print("After blurred:",compute_PSNR(blurred_image,image)) 
    