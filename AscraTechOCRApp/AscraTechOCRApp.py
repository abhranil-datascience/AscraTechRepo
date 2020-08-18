############################# Import Section ###########################################
from flask import Flask, request
from flask_restful import Resource, Api
import base64,cv2
import numpy as np
import pandas as pd
import pytesseract as pt
from pytesseract import Output
########################## Create Flask App ###########################################
app = Flask(__name__)
# creating an API object
api = Api(app)
######################### Common Functions #############################################
## Read Base64 string
def readb64(base64_string):
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
## Image Preprocessing Methods
#################### Image Preprocessing Type 1 ##################################
def PreprocessPANImageType1(CurrentImage):
    GrayScaleImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
    rgb_planes = cv2.split(GrayScaleImage)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    ConvertedImage=cv2.merge(result_planes)
    ConvertedImage=cv2.adaptiveThreshold(ConvertedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,20)
    return ConvertedImage
#################### Image Preprocessing Type 2 ##################################
def PreprocessPANImageType2(CurrentImage):
    GrayScaleImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
    ConvertedImage=cv2.adaptiveThreshold(GrayScaleImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,49)
    return ConvertedImage
#################### Image Preprocessing Type 3 ##################################
def PreprocessPANImageType3(CurrentImage):
    GrayScaleImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
    rgb_planes = cv2.split(GrayScaleImage)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    ConvertedImage=cv2.merge(result_planes)
    ConvertedImage=cv2.adaptiveThreshold(ConvertedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,81,17)
    return ConvertedImage
#################### Calculate Bottom ##############################
def CalculateBottom(row):
    current_top=row[2]
    current_height=row[3]
    return current_top+current_height
#################### Calculate Right ##############################
def CalculateRight(row):
    current_left=row[1]
    current_width=row[4]
    return current_left+current_width
################### OCR Function #####################################
def PerformOCR(GrayScaleImage):
    content=pt.image_to_data(GrayScaleImage,output_type=Output.DICT)
    Words=list(content['text'])
    left=list(content['left'])
    top=list(content['top'])
    width=list(content['width'])
    height=list(content['height'])
    content_dict=dict(Word=Words,Left=left,Top=top,Height=height,Width=width)
    res=pd.DataFrame.from_dict(content_dict)
    res=res[res['Word'].str.strip().str.len()>0]
    res['Word']=res['Word'].str.replace(",","")
    res=res[res['Word'].str.match(r'(^[a-zA-Z0-9/]*$)')==True]
    res['Word']=res['Word'].str.upper().str.strip()
    res['Bottom']=res.apply(func=CalculateBottom,axis=1)
    res['Right']=res.apply(func=CalculateRight,axis=1)
    res=res[['Word','Top','Left','Bottom','Right']]
    res=res.sort_values(by=['Top','Left'])
    return res
######### PAN Card OCR ##########
class PANCardOCR(Resource):
    def post(self):
        try:
            ################ Get File Name and Minimum Matches From Request ###############
            data = request.get_json()
            ImageData = data['ImageData']
            ################ Read Image from Base64 string ################################
            try:
                CurrentImage=readb64(ImageData)
            except:
                return {'Msg':'Error','Description':'Unable to convert base 64 string to Image'}
            ################ Preprocess Image #####################################
            try:
                PANCardImageProcessed1=PreprocessPANImageType1(CurrentImage)
                PANCardImageProcessed2=PreprocessPANImageType2(CurrentImage)
                PANCardImageProcessed3=PreprocessPANImageType3(CurrentImage)
            except:
                return {'Msg':'Error','Description':'Unable to preprocess Image'}
            #################### Perform OCR #####################################
            try:
                PANCardImageProcessed1DF=PerformOCR(PANCardImageProcessed1)
                PANCardImageProcessed2DF=PerformOCR(PANCardImageProcessed2)
                PANCardImageProcessed3DF=PerformOCR(PANCardImageProcessed3)
            except:
                return {'Msg':'Error','Description':'Corrupted Image - Unable to Perform OCR'}
            ########### Compare Dataframe to get common values ###################
            res=PANCardImageProcessed1DF[PANCardImageProcessed1DF['Word'].str.strip().isin(list(PANCardImageProcessed2DF['Word']))]
            ############### Fetch Region Of Interest #############################
            DepartmentRow=res[(res['Word'].str.contains("DEP")) | (res['Word'].str.contains("INC"))]
            if DepartmentRow.shape[0]!=0:
                DepartmentTopMax=DepartmentRow['Top'].max()
                newres=res[res['Top']>DepartmentTopMax+10]
            else:
                DepartmentRow=PANCardImageProcessed3DF[(PANCardImageProcessed3DF['Word'].str.contains("DEP")) | (PANCardImageProcessed3DF['Word'].str.contains("INC"))]
                GovRow=PANCardImageProcessed3DF[(PANCardImageProcessed3DF['Word'].str.contains("OF")) | (PANCardImageProcessed3DF['Word'].str.contains("IND"))]
                if (DepartmentRow.shape[0]!=0) and (GovRow.shape[0]!=0):
                    DepartmentTopMax=DepartmentRow['Top'].max()
                    GovRowLeftMin=GovRow['Left'].min()
                    newres=PANCardImageProcessed3DF[(PANCardImageProcessed3DF['Top']>DepartmentTopMax+10) & (PANCardImageProcessed3DF['Right']<GovRowLeftMin-150)]
                else:
                    return {'Msg':'Error','Description':'Poor Image Quality - Unable to Parse Tesseract OCR output'}
            ################ Fetch Name #########################################
            NameTop=list(newres['Top'])[0]
            NameTopUL=NameTop-20
            NameTopLL=NameTop+20
            WholeNameDF=newres[newres['Top'].between(NameTopUL,NameTopLL)]
            WholeNameDF=WholeNameDF.sort_values(by='Left')
            Name=" ".join(WholeNameDF['Word'])
            ############### Fetch DOB using "/" pattern ##########################
            DateOfBirth=""
            DateOfBirthDF=newres[newres['Word'].str.contains("/")]
            if len(list(DateOfBirthDF['Word']))!=0:
                DateOfBirth=list(DateOfBirthDF['Word'])[0]
            ############### Fetch Father's Name #################################
            NameBottom=max(list(WholeNameDF['Bottom']))
            if DateOfBirth!="":
                DateOfBirthTop=list(DateOfBirthDF['Top'])[0]
                FatherNameDF=newres[(newres['Top']>NameBottom+20) & (newres['Bottom']<DateOfBirthTop)]
                if FatherNameDF.shape[0]==0:
                    FatherNameDF=PANCardImageProcessed1DF[(PANCardImageProcessed1DF['Top']>NameBottom+10) & (PANCardImageProcessed1DF['Bottom']<DateOfBirthTop)]
            else:
                FatherNameDF=newres[(newres['Top']>NameBottom+10) & (newres['Bottom']<NameBottom+70)]
            FatherNameDF=FatherNameDF.sort_values(by='Left')
            FatherName=" ".join(list(FatherNameDF['Word']))
            ###### Try to Fetch DOB Again based on Father's Name if it's blank #######
            if DateOfBirth=="":
                DateOfBirthDF=PANCardImageProcessed1DF[PANCardImageProcessed1DF['Word'].str.contains("/")]
                if len(list(DateOfBirthDF['Word']))!=0:
                    DateOfBirth=list(DateOfBirthDF['Word'])[0]
            ################## Fetch PAN Number ###############################
            PANNumber=''
            PANNumberSeries=newres[(newres['Word'].str.match(r'^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$')==True) & (newres['Word'].str.len()==10)]['Word']
            if len(list(PANNumberSeries))!=0:
                PANNumber=list(PANNumberSeries)[0]
                PANNumber.upper()
            ############# Create Response Dict ###############################
            if (PANNumber!="") and (DateOfBirth!=""):
                ResponseDict=dict(Msg='Success',Name=Name,FatherName=FatherName,DateOfBirth=DateOfBirth,PANNumber=PANNumber)
            else:
                return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
            return ResponseDict
        except Exception as e:
            raise e
            return {'Msg':'Error','Description':'Unknown Exception Happened. Please make sure that the Image Orientation is upright.'}
#################### Configure URLs #########################
api.add_resource(PANCardOCR,'/PancardOCR')
#################  Run Flask Server ##########################
if __name__ == '__main__':
    app.run(debug = True)
