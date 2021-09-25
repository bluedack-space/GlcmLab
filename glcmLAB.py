import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

def plotGLCM(img, glcm_mean, glcm_std, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_asm, glcm_energy, glcm_max, glcm_entropy):
    mpl.rc('image', cmap='inferno')
    
    # plot                                                                      
    plt.figure(figsize=(10,4.5))

    fs     = 15
    plotid = [1,2,3,4,5,6,7,8,9,10]
    outs   = [img, glcm_mean, glcm_std,
              glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
              glcm_asm, glcm_energy, glcm_max,
              glcm_entropy]
    titles = ['original', 'mean', 'std',
              'contrast', 'dissimilarity', 'homogeneity',
              'ASM', 'energy', 'max',
              'entropy']
    imax = len(plotid)
    for i in range(imax):
        plt.subplot(2,(int)(imax/2),plotid[i])
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(outs[i])
        plt.title(titles[i], fontsize=fs)

    plt.tight_layout(pad=0.5)
    plt.savefig('output.png')
    plt.show()

def calcGLCM(img):
    kernel_size = 5
    levels      = 8
    symmetric   = False
    normed      = True

    # [01] Binarization
    img_bin = img//(256//levels) # [0:255]->[0:7]

    # [02] Calculate GLCM
    h,w       = img.shape

    glcm      = np.zeros ((h,w,levels,levels), dtype=np.uint8)
    kernel    = np.ones  ((kernel_size, kernel_size), np.uint8)
    img_bin_r = np.append(img_bin[:,1:], img_bin[:,-1:], axis=1)
    for i in range(levels):
        for j in range(levels):
            mask = (img_bin==i) & (img_bin_r==j)
            mask = mask.astype(np.uint8)
            glcm[:,:,i,j] = cv2.filter2D(mask, -1, kernel)

    glcm = glcm.astype(np.float32)

    if symmetric:
        glcm += glcm[:,:,::-1, ::-1]

    if normed:
        glcm  = glcm/glcm.sum(axis=(2,3), keepdims=True)

    #[03] martrix axis
    axis = np.arange(levels, dtype=np.float32)+1
    w    = axis.reshape(1,1,-1,1)
    x    = np.repeat(axis.reshape(1,-1), levels, axis=0)
    y    = np.repeat(axis.reshape(-1,1), levels, axis=1)

    # [04] Calculate GLCM Properties
    # GLCM contrast                                                             
    glcm_contrast      = np.sum(glcm*(x-y)**2, axis=(2,3))
    # GLCM dissimilarity                                                        
    glcm_dissimilarity = np.sum(glcm*np.abs(x-y), axis=(2,3))
    # GLCM homogeneity                                                          
    glcm_homogeneity   = np.sum(glcm/(1.0+(x-y)**2), axis=(2,3))
    # GLCM energy & ASM                                                         
    glcm_asm           = np.sum(glcm**2, axis=(2,3))
    # GLCM entropy                                                              
    ks                 = 5 # kernel_size                                                        
    pnorm              = glcm / np.sum(glcm, axis=(2,3), keepdims=True) + 1./ks**2
    glcm_entropy       = np.sum(-pnorm * np.log(pnorm), axis=(2,3))
    # GLCM mean                                                                 
    glcm_mean          = np.mean(glcm*w, axis=(2,3))
    # GLCM std                                                                  
    glcm_std           = np.std(glcm*w, axis=(2,3))
    # GLCM energy                                                               
    glcm_energy        = np.sqrt(glcm_asm)
    # GLCM max                                                                  
    glcm_max           = np.max(glcm, axis=(2,3))

    return glcm_mean, glcm_std, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_asm, glcm_energy, glcm_max, glcm_entropy

if __name__ == '__main__':
    #args          = sys.argv
    #fileNameImage = args[1]
    #img           = imageHandler.Open(fileNameImage,colorType="GRAYSCALE")

    capture = cv2.VideoCapture(0)
    for i in range(100):
        ret, frame = capture.read()
        print(ret)
        if ret==True:
            img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    img = img_gray
    #displayImage(img)
    glcm_mean, glcm_std, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_asm, glcm_energy, glcm_max, glcm_entropy = calcGLCM(img)
    plotGLCM (img, glcm_mean, glcm_std, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_asm, glcm_energy, glcm_max, glcm_entropy)
