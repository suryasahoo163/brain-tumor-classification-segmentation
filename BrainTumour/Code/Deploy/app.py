import tkinter as tk
from tkinter import filedialog
import ctypes
import cv2,os
from PIL import Image, ImageTk
import numpy as np
import time
from keras.models import load_model
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ctypes.windll.user32.SetProcessDPIAware()
cwd = os.chdir(r"C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\Code\Deploy")

# ===================================== Identifiers ========================================================
width = 1100
height = 800
brainImage = None
binaryPIL = None
segmentedPIL = None
selectedPIL = None
imageWidth =42
imageHeight = 14
predictedResult = None
globalBG = "#024652"
title = "Brain Tumor Detector and Predictor"
ClassLabels = ["Glioma Tumor","Meningioma Tumor","No Tumor","Pituitary Tumor"]



segmentationModel = load_model(r'C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\Models\ResUnet.keras')
classificationModel = load_model(r'C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\Models\brain_tumor_cnn.h5')
# ====================================== Functions ========================================================
def cv2_pil(image): # cv2 image
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((300, 256))
    image = ImageTk.PhotoImage(pil_image)
    return image # pil image

def predict():
    global brainImage, binaryPIL, segmentedPIL,selectedPIL, segmentationModel, predictedResult, classificationModel, loadingImage

    # image 1 actual image
    brainImage = upload_image() # actual uploaded image
    imageUpload["image"]=loadingImage
    addimagetext["text"] = "Loading..."
    imageUpload.update_idletasks()
    addimagetext.update_idletasks()
    time.sleep(2)
    # cv2 image for ml projection
    cv2image = cv2.cvtColor(cv2.resize(np.array(brainImage),(256,256)),cv2.COLOR_RGB2GRAY) # cv2 image with gray scale for prediction and segmentation 
    grayImage = cv2image.copy()
    brainImage = brainImage.resize((300,256))
    brainImage = ImageTk.PhotoImage(brainImage)
    # image 2 binary image
    binaryPIL = cv2_pil(cv2image)
     
    #### Prediction Part ======
   
    cv2image = cv2image.reshape(-1, 256, 256, 1)
    cv2image = cv2image / 255.0
    pred = segmentationModel.predict(cv2image)[0]
    pred_mask = (pred > 0.5).astype(np.uint8)

    mask_for_pil = pred_mask.reshape(256,256)*255

    # image 3 segmentade image
    segmentedPIL = cv2_pil(mask_for_pil)

    # image 4 for highlight the are of mask
    
    # Convert grayscale image to RGB so we can color it
    original_rgb = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2RGB)

    # Create a red overlay for tumor areas
    red_overlay = original_rgb.copy()
    red_overlay[mask_for_pil == 255] = [255, 0, 0]  # Red for tumor

    # OPTIONAL: Blend red overlay with original image (transparency effect)
    alpha = 0.5
    highlighted = cv2.addWeighted(red_overlay, alpha, original_rgb, 1 - alpha, 0)

    # Step 5: Convert to PIL Image
    selectedPIL = cv2_pil(highlighted)

    # class prediction 
    class_pred_index = classificationModel.predict(cv2image)[0].argmax()
    predictedResult = ClassLabels[class_pred_index]

    # tumor size  detection
    _, brain_mask = cv2.threshold(grayImage, thresh=30, maxval=255, type=cv2.THRESH_BINARY)
    brainSize = cv2.countNonZero(brain_mask)
   
    tumorSize = np.sum(pred_mask==1)
    ratioTB = 100*(tumorSize/brainSize)
    if class_pred_index == 2:
        # means no tumor
        selectedPIL = binaryPIL
        segmentedPIL = cv2_pil(np.zeros((256,256)))
        ratioTB = 0
 
    # image updation
    orignalImage["image"] = brainImage
    binaryImge["image"] = binaryPIL
    segmentPart["image"] = segmentedPIL
    highlightPart["image"] = selectedPIL
    resultText["text"] = f"Predicted Result:\n{predictedResult}"
    tumorsize["text"] = f"Size of the Tumor is\n{round(ratioTB,2)}%\nof the Brain"
    # frame updation
    uploadFrame.pack_forget()

    mainPage.pack(side="left",padx=40)
    predictPage.pack(side="right",padx=40)

def renderUploadPage():
    global plusImage
    imageUpload["image"] = plusImage
    addimagetext["text"] = "Upload Image"
    uploadFrame.pack(pady=100)
    mainPage.pack_forget()
    predictPage.pack_forget()

def upload_image():
    file = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file:
    
        image = Image.open(file)
        image = image.resize((256, 256))
    
    return image

# =================================== Window Root =========================================================
root = tk.Tk()
root.title(title)
root.geometry(f"{width}x{height}")
root.resizable(False,False)
root.configure(background=globalBG)

icon = Image.open("brain.ico")
icon = ImageTk.PhotoImage(icon)
root.iconphoto(False,icon)

plusImage = Image.open("plus.png")
plusImage = plusImage.resize((120,120)) 
plusImage = ImageTk.PhotoImage(plusImage)

iconPNG = Image.open("brain.png")
iconPNG = iconPNG.resize((45,45))
iconPNG = ImageTk.PhotoImage(iconPNG)

lensPNG = Image.open("lens.png")
lensPNG = lensPNG.resize((60,60))
lensPNG = ImageTk.PhotoImage(lensPNG)

loadingImage = Image.open("loading.png")
loadingImage = loadingImage.resize((120,120)) 
loadingImage = ImageTk.PhotoImage(loadingImage)

header = tk.Label(root,text=title,bg=globalBG,fg="#99f2ac",font=("Times New Roman",30,"bold"))
header.pack(pady=30)
headerImage = tk.Label(root,image= iconPNG,bg=globalBG).place(x=100,y=38)
headerImage2 = tk.Label(root,image= lensPNG,bg=globalBG).place(x=width - 60 - 100,y=28)

# ===================================== Imge Upload Page =================================================
uploadFrame = tk.Frame(root,width=1000,height=500,bg=globalBG)

imageUpload = tk.Button(uploadFrame,bg=globalBG,height = 120,width=120,border=0,image=plusImage,command=predict,activebackground=globalBG)
imageUpload.pack()

addimagetext = tk.Label(uploadFrame,bg=globalBG,text="Upload Image",fg="white",font=("Times New Roman",20,"bold"))
addimagetext.pack()

uploadFrame.pack(pady=100)
# ==================================== Predicted Image Display Page ======================================
mainPage = tk.Frame(root,bg=globalBG)
mainPage.pack_forget()

orignalImage = tk.Label(mainPage,bg=globalBG)
orignalImage.grid(row=0,column=0,padx=10,pady=10)

OILabel = tk.Label(mainPage,text="Original Image",bg=globalBG,fg="white",font=("Times New Roman",15,"bold"))
OILabel.grid(row=1,column=0,padx=10)

binaryImge = tk.Label(mainPage,bg=globalBG)
binaryImge.grid(row=0,column=1,padx=10,pady=10)

BILabel = tk.Label(mainPage,text="Binary Image",bg=globalBG,fg="white",font=("Times New Roman",15,"bold"))
BILabel.grid(row=1,column=1,padx=10)

segmentPart = tk.Label(mainPage,bg=globalBG)
segmentPart.grid(row=2,column=0,padx=10,pady=10)

SILabel = tk.Label(mainPage,text="Tumor Segment",bg=globalBG,fg="white",font=("Times New Roman",15,"bold"))
SILabel.grid(row=3,column=0,padx=10)

highlightPart = tk.Label(mainPage,bg=globalBG)
highlightPart.grid(row=2,column=1,padx=10,pady=10)

HILabel = tk.Label(mainPage,text="Tumor Highlighted",bg=globalBG,fg="white",font=("Times New Roman",15,"bold"))
HILabel.grid(row=3,column=1,padx=10)
# ===================================== Predicted Labels ================================================
predictPage = tk.Frame(root,bg=globalBG)
predictPage.pack_forget()

resultText = tk.Label(predictPage,bg=globalBG,fg="white",font=("Times New Roman",17,"bold"))
resultText.grid(column=0,row=0,pady=40,padx=25)

tumorsize = tk.Label(predictPage,bg=globalBG,fg="white",font=("Times New Roman",17,"bold")) # tumor size in percentage (tumor pixel count / total brain pixel count ) x 100
tumorsize.grid(row=1,column=0,pady=40,padx=25)

reupload = tk.Button(predictPage,text="Return to Upload",bg=globalBG,command= renderUploadPage,fg="white",font=("Times New Roman",17,"bold"),activebackground=globalBG,activeforeground="#ed221f")
reupload.grid(row=2,column=0,pady=40,padx=25)

root.mainloop()