############################# Import Section ###########################################
from flask import Flask, request
from flask_restful import Resource, Api
import base64,cv2,os
import numpy as np
import pandas as pd
import pytesseract as pt
from pytesseract import Output
import requests,random,string
from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.json_format import MessageToDict
import io,re
#from PIL import Image
########################## Create Flask App ###########################################
app = Flask(__name__)
# creating an API object
api = Api(app)
######################### Common Functions #############################################
## Image Preprocessing Methods
#################### Image Preprocessing Type 1 ##################################
def IncreaseIlluminationInImage(CurrentImage):
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
    return ConvertedImage
#################### PAN Image Preprocessing Type 2 ##################################
def PreprocessPANImageType1(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
    return ConvertedImage
#################### PAN Image Preprocessing Type 2 ##################################
def PreprocessPANImageType2(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,57,25)
    return ConvertedImage
#################### Aadhar Front Image Preprocessing Type 2 ##################################
def PreprocessAadharFrontImageType1(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,53,25)
    return ConvertedImage
#################### Aadhar Front Image Preprocessing Type 2 ##################################
def PreprocessAadharFrontImageType2(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,25)
    return ConvertedImage
#################### Aadhar Back Image Preprocessing Type 2 ##################################
def PreprocessAadharBackImageType1(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,83,15)
    return ConvertedImage
#################### Aadhar Back Image Preprocessing Type 2 ##################################
def PreprocessAadharBackImageType2(CurrentImage):
    ConvertedImage=cv2.adaptiveThreshold(CurrentImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,89,15)
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
def PerformPANOCRTesseract(GrayScaleImage):
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
def PerformAadharFrontOCRTesseract(GrayScaleImage):
    content=pt.image_to_data(GrayScaleImage,output_type=Output.DICT)
    Words=list(content['text'])
    left=list(content['left'])
    top=list(content['top'])
    width=list(content['width'])
    height=list(content['height'])
    content_dict=dict(Word=Words,Left=left,Top=top,Height=height,Width=width)
    res=pd.DataFrame.from_dict(content_dict)
    res=res[res['Word'].str.strip().str.len()>0]
    res['Bottom']=res.apply(func=CalculateBottom,axis=1)
    res['Right']=res.apply(func=CalculateRight,axis=1)
    res=res[['Word','Top','Left','Bottom','Right']]
    res['Word']=res['Word'].str.replace(",","")
    res=res[(res['Word'].str.match(r'(^[a-zA-Z0-9/:]*$)')==True)]
    res['Word']=res['Word'].str.upper().str.strip()
    #res=res.sort_values(by=['Top','Left'])
    return res
def PerformAadharBackOCRTesseract(GrayScaleImage):
    content=pt.image_to_data(GrayScaleImage,output_type=Output.DICT)
    Words=list(content['text'])
    left=list(content['left'])
    top=list(content['top'])
    width=list(content['width'])
    height=list(content['height'])
    content_dict=dict(Word=Words,Left=left,Top=top,Height=height,Width=width)
    res=pd.DataFrame.from_dict(content_dict)
    res=res[res['Word'].str.strip().str.len()>0]
    res['Bottom']=res.apply(func=CalculateBottom,axis=1)
    res['Right']=res.apply(func=CalculateRight,axis=1)
    res=res[['Word','Top','Left','Bottom','Right']]
    res['Word']=res['Word'].str.upper().str.strip()
    return res
def PerformPassportFrontOCRTesseract(GrayScaleImage):
    content=pt.image_to_data(GrayScaleImage,output_type=Output.DICT)
    Words=list(content['text'])
    left=list(content['left'])
    top=list(content['top'])
    width=list(content['width'])
    height=list(content['height'])
    content_dict=dict(Word=Words,Left=left,Top=top,Height=height,Width=width)
    ConvertedImageDF=pd.DataFrame.from_dict(content_dict)
    ConvertedImageDF['Bottom']=ConvertedImageDF.apply(func=CalculateBottom,axis=1)
    ConvertedImageDF['Right']=ConvertedImageDF.apply(func=CalculateRight,axis=1)
    ConvertedImageDF=ConvertedImageDF[['Word','Top','Left','Bottom','Right']]
    print("#################################################")
    print("")
    print("List of Word: ",list(ConvertedImageDF['Word']))
    print("")
    print("#################################################")
    return ConvertedImageDF
################## Clean OCR Data ####################################
def RemoveHindiCharacters(WordList):
    TempWordList=[]
    for word in WordList:
        ValidCharacters = [c for c in word if ord(c) < 128]
        TempWordList.append("".join(ValidCharacters).strip())
    return TempWordList
def CleanText(WordList):
    TempWordList=[]
    for word in WordList:
        if "/" in word:
            if word.replace("/","").isalpha():
                TempWordList.append(word.replace("/",""))
            else:
                TempWordList.append(word)
        else:
            TempWordList.append(word)
    return TempWordList
def CleanPassportFrontData(CurrentDF):
    CurrentDF=CurrentDF[CurrentDF['Word'].str.strip().str.len()>0]
    CurrentDF['Word']=CurrentDF['Word'].str.replace(",","")
    CurrentDF['Word']=RemoveHindiCharacters(list(CurrentDF['Word']))
    CurrentDF=CurrentDF[(CurrentDF['Word'].str.match(r'(^[a-zA-Z0-9/:<.]*$)')==True)]
    CurrentDF['Word']=CurrentDF['Word'].str.upper().str.strip()
    CurrentDF['Word']=CleanText(list(CurrentDF['Word']))
    CurrentDF = CurrentDF[CurrentDF['Word']!="/"]
    CurrentDF = CurrentDF[CurrentDF['Word']!=""]
    return CurrentDF
################### Function to Reform Google Vision API Dataframe ###################
def CreateTop(row):
    current_uly=row[8]
    current_ury=row[6]
    top=min(current_uly,current_ury)
    return top
def CreateBottom(row):
    current_lly=row[2]
    current_lry=row[4]
    bottom=max(current_lly,current_lry)
    return bottom
def CreateLeft(row):
    current_llx=row[1]
    current_ulx=row[7]
    left=min(current_llx,current_ulx)
    return left
def CreateRight(row):
    current_lrx=row[3]
    current_urx=row[5]
    right=max(current_lrx,current_urx)
    return right
############## OCR Using Google Vision API ##########################
def PerformOCRGoogleVisionAPI(current_input_file_path):
    with io.open(current_input_file_path, 'rb') as gen_image_file:
        content = gen_image_file.read()
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.types.Image(content=content)
        response = client.text_detection(image=image)
        DictResponse=MessageToDict(response)
        WordsAndCoordinates=DictResponse['textAnnotations'][1:]
        word_list=[]
        llx_list=[]
        lly_list=[]
        lrx_list=[]
        lry_list=[]
        urx_list=[]
        ury_list=[]
        ulx_list=[]
        uly_list=[]
        for i in range(0,len(WordsAndCoordinates)):
            word_list.append(WordsAndCoordinates[i]['description'])
            llx_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][0]['x'])
            lly_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][0]['y'])
            lrx_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][1]['x'])
            lry_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][1]['y'])
            urx_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][2]['x'])
            ury_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][2]['y'])
            ulx_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][3]['x'])
            uly_list.append(WordsAndCoordinates[i]['boundingPoly']['vertices'][3]['y'])
        ##################### Create Dictionary for the lists #####################
        WordsAndCoordinatesDict={"Word":word_list,'llx':llx_list,'lly':lly_list,'lrx':lrx_list,'lry':lry_list,'urx':urx_list,'ury':ury_list,'ulx':ulx_list,'uly':uly_list}
        ####################### Create Dataframe ######################
        WordsAndCoordinatesDF = pd.DataFrame.from_dict(WordsAndCoordinatesDict)
        print(list(WordsAndCoordinatesDF['Word']))
        return WordsAndCoordinatesDF
    except:
        return "Error"
################### Function to fetch Valid Values #####################################
def GetValidValues(CurrentDF):
    ValidValues=[]
    for ind in CurrentDF.index:
        if ind == 0:
            ValidValues.append(CurrentDF['Word'][ind])
        else:
            current_left = CurrentDF['Left'][ind]
            previous_right = CurrentDF['Right'][ind-1]
            if current_left - previous_right > 60:
                break
            else:
                ValidValues.append(CurrentDF['Word'][ind])
    return " ".join(ValidValues).strip()
######### PAN Card OCR ##########
class PANCardOCR(Resource):
    def post(self):
        try:
            ################ Get File Name and Minimum Matches From Request ###############
            data = request.get_json()
            ImageFile = data['ImageFile']
            FileType=data['filetype']
            DownloadDirectory="/mnt/tmp"
            randomfivedigitnumber=random.randint(10000,99999)
            letters = string.ascii_lowercase
            randomfivecharacters=''.join(random.choice(letters) for i in range(5))
            if FileType.lower()=="jpg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpg"
            elif FileType.lower()=="jpeg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpeg"
            elif FileType.lower()=="png":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".png"
            else:
                return{'msg':'Error','description':'Unsupported File Extension'}
            DownloadFilePath=DownloadDirectory+"/"+FileName
            ################## Download File #######################
            try:
                response=requests.get(str(ImageFile))
                if response.status_code != 200:
                    return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            except:
                return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            ############# Write downloaded file to local ##########
            try:
                with open(DownloadFilePath,'wb') as f:
                    f.write(response.content)
            except:
                return{'msg':'Error','description':'Unable to save downloaded file.'}
            ################ Read Image from Base64 string ################################
            try:
                CurrentImage=cv2.imread(DownloadFilePath)
                #os.remove(DownloadFilePath)
            except:
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to read downladed image.'}
            ################ Preprocess Image #####################################
            try:
                IlluminatedPANCard=IncreaseIlluminationInImage(CurrentImage)
                PANCardImageProcessed1=PreprocessPANImageType1(IlluminatedPANCard)
                PANCardImageProcessed2=PreprocessPANImageType2(IlluminatedPANCard)
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to preprocess Image'}
            #################### Perform OCR #####################################
            try:
                PANCardImageProcessed1DF=PerformPANOCRTesseract(PANCardImageProcessed1)
                PANCardImageProcessed2DF=PerformPANOCRTesseract(PANCardImageProcessed2)
                PANCardImageProcessed1DF=PANCardImageProcessed1DF[PANCardImageProcessed1DF['Word'].isin(list(PANCardImageProcessed2DF['Word']))]
                PANCardImageProcessed1DF=PANCardImageProcessed1DF.reset_index(drop=True)
                res=PANCardImageProcessed1DF.copy()
                DepartmentRow=res[(res['Word'].str.lower().str.contains("dep")) | (res['Word'].str.lower().str.contains("inc")) | (res['Word'].str.lower().str.contains("gov")) | (res['Word'].str.lower().str.contains("indi"))]
                if DepartmentRow.shape[0]!=0:
                    DepartmentTop=DepartmentRow['Bottom'].max()
                    newres=res[res['Top']>DepartmentTop]
                else:
                    newres = res
                GovtRow=res[(res['Word'].str.lower().str.contains("gov")) | (res['Word'].str.lower().str.contains("ind")) | (res['Word'].str.lower().str.contains("of"))]
                if GovtRow.shape[0]!=0:
                    GovtRowLeft=GovtRow['Left'].min()
                    newres=newres[newres['Right']<GovtRowLeft]
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Corrupted Image - Unable to Perform OCR'}
            try:
                ################ Fetch Name #########################################
                Name=""
                NameTop=list(newres['Top'])
                if len(NameTop)!=0:
                    NameTop=NameTop[0]
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
                FatherName=""
                if Name != "":
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
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to fetch details from Tesseract Response'}
            ############# Create Response Dict ###############################
            if ((PANNumber=="") and (DateOfBirth=="")):
                ################# Since Tesseract Failed So Calling Google Vision API ######################################
                try:
                    ################ Get Dataframe from Google Vision API ######################
                    WordsAndCoordinatesDF=PerformOCRGoogleVisionAPI(DownloadFilePath)
                    os.remove(DownloadFilePath)
                    ################ Check Response from Google Vision API ######################
                    if str(type(WordsAndCoordinatesDF)) != "<class 'pandas.core.frame.DataFrame'>":
                        return {'Msg':'Error','Description':'Unable to Perform OCR using Google Vision API - Poor Image Quality.'}
                    else:
                        try:
                            ################ Filter Dataframe ######################
                            res=WordsAndCoordinatesDF.copy()
                            res=res[res['Word'].str.match(r"^[A-Za-z0-9/]*$")==True]
                            DepartmentRow=res[(res['Word'].str.lower().str.contains("dep")) | (res['Word'].str.lower().str.contains("inc")) | (res['Word'].str.lower().str.contains("gov")) | (res['Word'].str.lower().str.contains("indi"))]
                            if DepartmentRow.shape[0]!=0:
                                DepartmentTop=DepartmentRow['lly'].max()
                                newres=res[res['lly']>DepartmentTop+10]
                            ############# Fetch Name #####################
                            Name=""
                            NameTop=list(newres['uly'])
                            if len(NameTop)!=0:
                                NameTop=NameTop[0]
                                NameTopUL=NameTop-10
                                NameTopLL=NameTop+10
                                WholeNameDF=newres[newres['uly'].between(NameTopUL,NameTopLL)]
                                WholeNameDF=WholeNameDF.sort_values(by='ulx')
                                Name=" ".join(WholeNameDF['Word'])
                            ############# Fetch Date Of Birth #####################
                            DateOfBirth=""
                            DateOfBirthDF=newres[newres['Word'].str.contains("/")]
                            if len(list(DateOfBirthDF['Word']))!=0:
                                DateOfBirth=list(DateOfBirthDF['Word'])[0]
                            ############# Fetch Father's Name #####################
                            FatherName=""
                            if Name!="":
                                NameBottom=max(list(WholeNameDF['lly']))
                                if DateOfBirth!="":
                                    DateOfBirthTop=list(DateOfBirthDF['uly'])[0]
                                    FatherNameDF=newres[(newres['lly']>NameBottom+15) & (newres['uly']<DateOfBirthTop)]
                                    if FatherNameDF.shape[0]!=0:
                                        FatherNameDF=FatherNameDF.sort_values(by='ulx')
                                        FatherName=" ".join(FatherNameDF['Word'])
                                else:
                                    FatherNameDF=newres[(newres['uly']>NameBottom+10) & (newres['lly']<NameBottom+70)]
                                FatherNameDF=FatherNameDF.sort_values(by='ulx')
                                FatherName=" ".join(list(FatherNameDF['Word']))
                            ############# Fetch PAN Number #####################
                            PANNumber=''
                            PANNumberSeries=newres[(newres['Word'].str.match(r'^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$')==True) & (newres['Word'].str.len()==10)]['Word']
                            if len(list(PANNumberSeries))!=0:
                                PANNumber=list(PANNumberSeries)[0]
                                PANNumber.upper()
                            ############### Create Response Dict #######################
                            print("Name: ",Name," || Father's Name: ",FatherName," || DateOfBirth: ",DateOfBirth," || PANNumber: ",PANNumber)
                            if ((PANNumber=="") and (DateOfBirth=="")):
                                return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
                            else:
                                ResponseDict=dict(Msg='Success',Name=Name,FatherName=FatherName,DateOfBirth=DateOfBirth,PANNumber=PANNumber,Method="GoogleVisionAPI")
                                return ResponseDict
                        except Exception as e:
                            print(e)
                            return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
            else:
                os.remove(DownloadFilePath)
                print("Name: ",Name," || Father's Name: ",FatherName," || DateOfBirth: ",DateOfBirth," || PANNumber: ",PANNumber)
                ResponseDict=dict(Msg='Success',Name=Name,FatherName=FatherName,DateOfBirth=DateOfBirth,PANNumber=PANNumber,Method="Tesseract")
                return ResponseDict
        except Exception as e:
            print(e)
            os.remove(DownloadFilePath)
            return {'Msg':'Error','Description':'Unknown Exception Happened. Please make sure that the Image Orientation is upright.'}
class AadharFrontOCR(Resource):
    def post(self):
        try:
            ################ Get File Name and Minimum Matches From Request ###############
            data = request.get_json()
            ImageFile = data['ImageFile']
            FileType=data['filetype']
            DownloadDirectory="/mnt/tmp"
            randomfivedigitnumber=random.randint(10000,99999)
            letters = string.ascii_lowercase
            randomfivecharacters=''.join(random.choice(letters) for i in range(5))
            if FileType.lower()=="jpg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpg"
            elif FileType.lower()=="jpeg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpeg"
            elif FileType.lower()=="png":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".png"
            else:
                return{'msg':'Error','description':'Unsupported File Extension'}
            DownloadFilePath=DownloadDirectory+"/"+FileName
            ################## Download File #######################
            try:
                response=requests.get(str(ImageFile))
                if response.status_code != 200:
                    return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            except:
                return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            ############# Write downloaded file to local ##########
            try:
                with open(DownloadFilePath,'wb') as f:
                    f.write(response.content)
            except:
                return{'msg':'Error','description':'Unable to save downloaded file.'}
            ################ Read Image from Base64 string ################################
            try:
                CurrentImage=cv2.imread(DownloadFilePath)
                CurrentImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2RGB)
                CurrentImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
                #os.remove(DownloadFilePath)
            except:
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to read downladed image.'}
            ################ Preprocess Image #####################################
            try:
                AadharCardImageProcessed1=PreprocessAadharFrontImageType1(CurrentImage)
                AadharCardImageProcessed2=PreprocessAadharFrontImageType2(CurrentImage)
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to preprocess Image'}
            #################### Perform OCR #####################################
            try:
                AadharCardImageProcessed1DF=PerformAadharFrontOCRTesseract(AadharCardImageProcessed1)
                AadharCardImageProcessed2DF=PerformAadharFrontOCRTesseract(AadharCardImageProcessed2)
                ConvertedImageDF=AadharCardImageProcessed2DF[AadharCardImageProcessed2DF['Word'].str.strip().isin(list(AadharCardImageProcessed1DF['Word']))]
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Corrupted Image - Unable to Perform OCR'}
            ################ Fetch Birth Year #####################################
            try:
                BirthYear=""
                BirthYearDF=ConvertedImageDF[ConvertedImageDF['Word'].isin(['YEAR','BIRTH'])]
                if BirthYearDF.shape[0]!=0:
                    BirthYearTop=BirthYearDF['Top'].min()-15
                    BirthYearBottom=BirthYearDF['Bottom'].max()+15
                    BirthYearDF=ConvertedImageDF[(ConvertedImageDF['Top']>=BirthYearTop) & (ConvertedImageDF['Bottom']<=BirthYearBottom)]
                    BirthYearDF=BirthYearDF[BirthYearDF['Word'].str.match(r'(^[0-9]*$)')==True]
                    if BirthYearDF.shape[0]!=0:
                        BirthYear="".join(BirthYearDF['Word'])
                if BirthYear == "":
                    test="".join(list(ConvertedImageDF['Word']))
                    MatchList=re.findall(r"\d\d\d\d",test)
                    MatchListAadhar=re.findall(r"\d\d\d\d\d\d\d\d\d\d\d\d",test)
                    if len(MatchList)!=0:
                        ProbableBirthYear=MatchList[0]
                        if len(MatchListAadhar) != 0:
                            if ProbableBirthYear not in MatchListAadhar[0]:
                                BirthYear=ProbableBirthYear
                        else:
                            BirthYear=ProbableBirthYear
                if (BirthYear != "") and (BirthYearDF.shape[0]==0):
                    BirthYearDF=ConvertedImageDF[ConvertedImageDF['Word'].isin([BirthYear])]
                ################## Fetch Sex ################################################
                Sex=""
                AllWords=list(ConvertedImageDF['Word'])
                if "MALE" in AllWords:
                    Sex="Male"
                else:
                    Sex="Female"
                ################## Fetch Aadhar Number #####################################
                AadharNumber=""
                for word in list(ConvertedImageDF['Word']):
                    if re.match(r"\d\d\d\d\d\d\d\d\d\d\d\d",word):
                        AadharNumber=word
                        break
                UniqueTops=list(ConvertedImageDF['Top'].unique())
                ValidUniqueTops=[]
                for top in UniqueTops:
                    current_top_range=[]
                    for i in range(-10,11):
                        current_top_range.append(top+i)
                    df_records=ConvertedImageDF[ConvertedImageDF['Top'].isin(current_top_range)]
                    if df_records.shape[0]>1:
                        ValidUniqueTops.append(top)
                ConvertedImageDF=ConvertedImageDF[ConvertedImageDF['Top'].isin(ValidUniqueTops)]
                if AadharNumber == "":
                    test="".join(list(ConvertedImageDF['Word']))
                    if BirthYear!="":
                        test=test.replace(BirthYear,'')
                    MatchList=re.findall(r"\d\d\d\d\d\d\d\d\d\d\d\d",test)
                    if len(MatchList) != 0:
                        AadharNumber=MatchList[0]

                ################## Fetch Name #####################################
                Name=""
                if "GUARDIAN" in list(ConvertedImageDF['Word']):
                    GUARDIANDF=ConvertedImageDF[ConvertedImageDF['Word']=="GUARDIAN"]
                    GUARDIANTop=list(GUARDIANDF['Top'])[0]-60
                    NameDF=ConvertedImageDF[ConvertedImageDF['Top']<GUARDIANTop].tail(1)
                    if NameDF.shape[0]==1:
                        NameDFTop=list(NameDF['Top'])[0]
                        NameDFBottom=list(NameDF['Bottom'])[0]
                        NameDF=ConvertedImageDF[(ConvertedImageDF['Top']>=NameDFTop-20) & (ConvertedImageDF['Bottom']<=NameDFBottom+20)]
                        NameDF=NameDF.sort_values(by='Left')
                        Name=" ".join(NameDF['Word'])
                if Name == "":
                    if ("FATHER" in list(ConvertedImageDF['Word'])) or ("FATHER:" in list(ConvertedImageDF['Word'])):
                        FatherDF=ConvertedImageDF[ConvertedImageDF['Word'].isin(["FATHER:",'FATHER'])]
                        FatherTop=FatherDF['Top'].min()
                        NameDF=ConvertedImageDF[ConvertedImageDF['Top']<FatherTop-60].tail(1)
                        if NameDF.shape[0]==1:
                            NameDFTop=list(NameDF['Top'])[0]
                            NameDFBottom=list(NameDF['Bottom'])[0]
                            NameDF=ConvertedImageDF[(ConvertedImageDF['Top']>=NameDFTop-20) & (ConvertedImageDF['Bottom']<=NameDFBottom+20)]
                            NameDF=NameDF.sort_values(by='Left')
                            Name=" ".join(NameDF['Word'])
                if Name == "":
                    if BirthYearDF.shape[0]!=0:
                        BirthYearTop=BirthYearDF['Top'].min()-40
                        NameDF=ConvertedImageDF[ConvertedImageDF['Top']<BirthYearTop].tail(1)
                        if NameDF.shape[0]==1:
                            NameDFTop=list(NameDF['Top'])[0]
                            NameDFBottom=list(NameDF['Bottom'])[0]
                            NameDF=ConvertedImageDF[(ConvertedImageDF['Top']>=NameDFTop-20) & (ConvertedImageDF['Bottom']<=NameDFBottom+20)]
                            NameDF=NameDF.sort_values(by='Left')
                            Name=" ".join(NameDF['Word'])
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to fetch details from Tesseract Response'}
            ############# Create Response Dict ###############################
            if Name=="" or AadharNumber=="" or BirthYear=="":
                ################# Since Tesseract Failed So Calling Google Vision API ######################################
                try:
                    ################ Get Dataframe from Google Vision API ######################
                    WordsAndCoordinatesDF=PerformOCRGoogleVisionAPI(DownloadFilePath)
                    os.remove(DownloadFilePath)
                    ################ Check Response from Google Vision API ######################
                    if str(type(WordsAndCoordinatesDF)) != "<class 'pandas.core.frame.DataFrame'>":
                        return {'Msg':'Error','Description':'Unable to Perform OCR using Google Vision API - Poor Image Quality.'}
                    else:
                        try:
                            WordsAndCoordinatesDF['Top']=WordsAndCoordinatesDF.apply(func=CreateTop,axis=1)
                            WordsAndCoordinatesDF['Bottom']=WordsAndCoordinatesDF.apply(func=CreateBottom,axis=1)
                            WordsAndCoordinatesDF['Left']=WordsAndCoordinatesDF.apply(func=CreateLeft,axis=1)
                            WordsAndCoordinatesDF['Right']=WordsAndCoordinatesDF.apply(func=CreateRight,axis=1)
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[['Word','Top','Bottom','Left','Right']]
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Word'].str.match(r'(^[a-zA-Z0-9]*$)')==True)]
                        except Exception as e:
                            print(e)
                            return {'Msg':'Error','Description':'Unable to reform Vision API Dataframe'}
                        try:
                            #################### Fetch Birth Year ###################################
                            BirthYear=""
                            BirthDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word']=="Birth"]
                            BirthRight=list(BirthDF['Right'])[0]
                            BirthTop=list(BirthDF['Top'])[0]
                            BirthBottom=list(BirthDF['Bottom'])[0]
                            BirthYear=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Top']>BirthTop-20) & (WordsAndCoordinatesDF['Bottom']<BirthBottom+20) & (WordsAndCoordinatesDF['Left']>BirthRight)]['Word']
                            BirthYear=" ".join(BirthYear)
                            ################ Fetch Aadhar Number ####################################
                            AadharNumber=""
                            AadharCardDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].str.match(r'(\d\d\d\d)')==True]
                            if AadharCardDF.shape[0]>1:
                                AadharCardDF=AadharCardDF[AadharCardDF['Word']!=BirthYear]
                                AadharCardDF=AadharCardDF.sort_values(by='Left')
                                AadharNumber="".join(AadharCardDF['Word'])
                            ####################### Fetch Sex #######################################
                            Sex=""
                            AllWords=list(WordsAndCoordinatesDF['Word'].str.lower())
                            if "male" in AllWords:
                                Sex="Male"
                            else:
                                Sex="Female"
                            ######################## Fetch Name #####################################
                            Name=""
                            if "GUARDIAN" in list(WordsAndCoordinatesDF['Word'].str.upper()):
                                GUARDIANDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].str.upper()=="GUARDIAN"]
                                GUARDIANTop=list(GUARDIANDF['Top'])[0]-60
                                NameDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Top']<GUARDIANTop].tail(1)
                                if NameDF.shape[0]==1:
                                    NameDFTop=list(NameDF['Top'])[0]
                                    NameDFBottom=list(NameDF['Bottom'])[0]
                                    NameDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Top']>=NameDFTop-40) & (WordsAndCoordinatesDF['Bottom']<=NameDFBottom+20)]
                                    NameDF=NameDF.sort_values(by='Left')
                                    Name=" ".join(NameDF['Word'])
                            if Name == "":
                                if ("FATHER" in list(WordsAndCoordinatesDF['Word'].str.upper())) or ("FATHER:" in list(WordsAndCoordinatesDF['Word'].str.upper())):
                                    FatherDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].str.upper().isin(["FATHER:",'FATHER'])]
                                    FatherTop=FatherDF['Top'].min()
                                    NameDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Top']<FatherTop-60].tail(1)
                                    if NameDF.shape[0]==1:
                                        NameDFTop=list(NameDF['Top'])[0]
                                        NameDFBottom=list(NameDF['Bottom'])[0]
                                        NameDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Top']>=NameDFTop-20) & (WordsAndCoordinatesDF['Bottom']<=NameDFBottom+20)]
                                        NameDF=NameDF.sort_values(by='Left')
                                        Name=" ".join(NameDF['Word'])
                            if Name == "":
                                if BirthDF.shape[0]!=0:
                                    BirthYearTop=BirthDF['Top'].min()-40
                                    NameDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Top']<BirthYearTop].tail(1)
                                    if NameDF.shape[0]==1:
                                        NameDFTop=list(NameDF['Top'])[0]
                                        NameDFBottom=list(NameDF['Bottom'])[0]
                                        NameDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Top']>=NameDFTop-20) & (WordsAndCoordinatesDF['Bottom']<=NameDFBottom+20)]
                                        NameDF=NameDF.sort_values(by='Left')
                                        Name=" ".join(NameDF['Word'])
                            ############### Create Response Dict #######################
                            if (Name!="") and (AadharNumber!=""):
                                ResponseDict=dict(Msg='Success',Name=Name,AadharNumber=AadharNumber,Sex=Sex,BirthYear=BirthYear,Method="GoogleVisionAPI")
                                return ResponseDict
                            else:
                                return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
                        except Exception as e:
                            print(e)
                            return {'Msg':'Error','Description':'Unable to fetch data from Vision API output.'}
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
            else:
                os.remove(DownloadFilePath)
                print("Name: ",Name," || AadharNumber: ",AadharNumber," || Sex: ",Sex," || BirthYear: ",BirthYear)
                ResponseDict=dict(Msg='Success',Name=Name,AadharNumber=AadharNumber,Sex=Sex,BirthYear=BirthYear,Method="Tesseract")
                return ResponseDict
        except Exception as e:
            print(e)
            os.remove(DownloadFilePath)
            return {'Msg':'Error','Description':'Unknown Exception Happened. Please make sure that the Image Orientation is upright.'}
class AadharBackOCR(Resource):
    def post(self):
        ################ Get File Name and Minimum Matches From Request ###############
        try:
            data = request.get_json()
            ImageFile = data['ImageFile']
            FileType=data['filetype']
            DownloadDirectory="/mnt/tmp"
            randomfivedigitnumber=random.randint(10000,99999)
            letters = string.ascii_lowercase
            randomfivecharacters=''.join(random.choice(letters) for i in range(5))
            if FileType.lower()=="jpg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpg"
            elif FileType.lower()=="jpeg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpeg"
            elif FileType.lower()=="png":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".png"
            else:
                return{'msg':'Error','description':'Unsupported File Extension'}
            DownloadFilePath=DownloadDirectory+"/"+FileName
            ################## Download File #######################
            try:
                response=requests.get(str(ImageFile))
                if response.status_code != 200:
                    return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            except:
                return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            ############# Write downloaded file to local ##########
            try:
                with open(DownloadFilePath,'wb') as f:
                    f.write(response.content)
            except:
                return{'msg':'Error','description':'Unable to save downloaded file.'}
            ################ Read Image from Base64 string ################################
            try:
                CurrentImage=cv2.imread(DownloadFilePath)
                #os.remove(DownloadFilePath)
            except:
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to read downladed image.'}
            ################ Preprocess Image #####################################
            try:
                CurrentImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2RGB)
                CurrentImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
                ConvertedImage1=PreprocessAadharBackImageType1(CurrentImage)
                ConvertedImage2=PreprocessAadharBackImageType2(CurrentImage)
            except:
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to preprocess Image.'}
            #################### Perform OCR #####################################
            try:
                AadharCardImageProcessed1DF=PerformAadharBackOCRTesseract(ConvertedImage1)
                AadharCardImageProcessed2DF=PerformAadharBackOCRTesseract(ConvertedImage2)
                ConvertedImageDF=AadharCardImageProcessed2DF[AadharCardImageProcessed2DF['Word'].str.strip().isin(list(AadharCardImageProcessed1DF['Word']))]
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Corrupted Image - Unable to Perform OCR'}
            ################ Fetch Address #####################################
            Address=""
            try:
                if ("ADDRESS:" in list(ConvertedImageDF['Word'])) or ("ADDRESS" in list(ConvertedImageDF['Word'])):
                    AddressLeft=ConvertedImageDF[ConvertedImageDF['Word'].isin(["ADDRESS:","ADDRESS"])]['Left'].min()
                    AddressLeft
                    AddressTop=ConvertedImageDF[ConvertedImageDF['Word'].isin(["ADDRESS:","ADDRESS"])]['Top'].max()
                    AddressTop
                    ConvertedImageDF=ConvertedImageDF[(ConvertedImageDF['Left']>=AddressLeft-5) & (ConvertedImageDF['Top']>=AddressTop-20)]
                    LowerLimitDF=ConvertedImageDF[ConvertedImageDF['Word'].isin(['BOX','1947','1800','HELP@UIDAI.GOV.IN','WWW.ULDAL.GOV.IN','P.O.'])]
                    LowerLimit=LowerLimitDF['Top'].min()
                    ConvertedImageDF=ConvertedImageDF[ConvertedImageDF['Bottom']<LowerLimit-60]
                    Address=" ".join(ConvertedImageDF['Word'])
                    Address=Address.replace("ADDRESS:","")
                    Address=Address.replace("ADDRESS","").strip()
                    if Address != "":
                        os.remove(DownloadFilePath)
                        ResponseDict=dict(Msg='Success',Address=Address,Method="Tesseract")
                        return ResponseDict
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to fetch details from Tesseract Response'}
            if Address == "":
                ################# Since Tesseract Failed So Calling Google Vision API ######################################
                try:
                    ################ Get Dataframe from Google Vision API ######################
                    WordsAndCoordinatesDF=PerformOCRGoogleVisionAPI(DownloadFilePath)
                    os.remove(DownloadFilePath)
                    ################ Check Response from Google Vision API ######################
                    if str(type(WordsAndCoordinatesDF)) != "<class 'pandas.core.frame.DataFrame'>":
                        return {'Msg':'Error','Description':'Unable to Perform OCR using Google Vision API - Poor Image Quality.'}
                    else:
                        try:
                            WordsAndCoordinatesDF['Top']=WordsAndCoordinatesDF.apply(func=CreateTop,axis=1)
                            WordsAndCoordinatesDF['Bottom']=WordsAndCoordinatesDF.apply(func=CreateBottom,axis=1)
                            WordsAndCoordinatesDF['Left']=WordsAndCoordinatesDF.apply(func=CreateLeft,axis=1)
                            WordsAndCoordinatesDF['Right']=WordsAndCoordinatesDF.apply(func=CreateRight,axis=1)
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[['Word','Top','Bottom','Left','Right']]
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Word'].str.match(r'(^[a-zA-Z0-9]*$)')==True)]
                        except Exception as e:
                            print(e)
                            return {'Msg':'Error','Description':'Unable to reform Vision API Dataframe'}
                        #################### Fetch Address ###################################
                        Address=""
                        if ("ADDRESS:" in list(WordsAndCoordinatesDF['Word'])) or ("ADDRESS" in list(WordsAndCoordinatesDF['Word'])):
                            AddressLeft=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].isin(["ADDRESS:","ADDRESS"])]['Left'].min()
                            AddressTop=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].isin(["ADDRESS:","ADDRESS"])]['Top'].max()
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[(WordsAndCoordinatesDF['Left']>=AddressLeft-5) & (WordsAndCoordinatesDF['Top']>=AddressTop-20)]
                            LowerLimitDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Word'].isin(['BOX','1947','1800','HELP@UIDAI.GOV.IN','WWW.ULDAL.GOV.IN','P.O.'])]
                            LowerLimit=LowerLimitDF['Top'].min()
                            WordsAndCoordinatesDF=WordsAndCoordinatesDF[WordsAndCoordinatesDF['Bottom']<LowerLimit-150]
                            Address=" ".join(WordsAndCoordinatesDF['Word'])
                            Address=Address.replace("ADDRESS:","")
                            Address=Address.replace("ADDRESS","").strip()
                        if Address!="":
                            ResponseDict=dict(Msg='Success',Address=Address,Method="GoogleVisionAPI")
                            return ResponseDict
                        else:
                            return {'Msg':'Error','Description':'Unable to Perform OCR - Poor Image Quality.'}
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to fetch data from Vision API output.'}
        except Exception as e:
            print(e)
            return {'Msg':'Error','Description':'Unknown Exception Happened. Please make sure that the Image Orientation is upright.'}
class PassportFrontOCR(Resource):
    def post(self):
        try:
            ############### Initialize Variables ###########################
            PassportNumber=""
            Surname=""
            GivenName=""
            Nationality=""
            Sex=""
            DateOfBirth=""
            PlaceOfBirth=""
            PlaceOfIssue=""
            DateOfIssue=""
            DateOfExpiry=""
            ListOfPlaceOfIssue = ['AHMEDABAD','AMRITSAR','BANGALORE','BAREILLY','BHOPAL','BHUBANESWAR','CHANDIGARH','CHENNAI',
            'COIMBATORE','CUDDALORE','DEHRADUN','DELHI','DHULE','GHAZIABAD','GUWAHATI','HYDERABAD','JAIPUR','JALANDHAR','JAMMU','KOCHI','KOLKATA',
            'KOZHIKODE','LUCKNOW','MADURAI','MALAPPURAM','MUMBAI','NAGPUR','PANAJI','PATNA','PUNE','RAIPUR','RANCHI','SHIMLA','SRINAGAR','SURAT',
            'THANE','THIRUVANANTHAPURAM','TIRUCHIRAPPALLI','VISAKHAPATNAM']
            ################ Get File Name and Minimum Matches From Request ###############
            data = request.get_json()
            ImageFile = data['ImageFile']
            FileType=data['filetype']
            DownloadDirectory="/mnt/tmp"
            randomfivedigitnumber=random.randint(10000,99999)
            letters = string.ascii_lowercase
            randomfivecharacters=''.join(random.choice(letters) for i in range(5))
            if FileType.lower()=="jpg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpg"
            elif FileType.lower()=="jpeg":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".jpeg"
            elif FileType.lower()=="png":
                FileName="File_"+str(randomfivedigitnumber)+"_"+randomfivecharacters+".png"
            else:
                return{'msg':'Error','description':'Unsupported File Extension'}
            DownloadFilePath=DownloadDirectory+"/"+FileName
            ################## Download File #######################
            try:
                response=requests.get(str(ImageFile))
                if response.status_code != 200:
                    return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            except:
                return{'msg':'Error','description':'Unable to download file. Please check the file url and permissions again.'}
            ############# Write downloaded file to local ##########
            try:
                with open(DownloadFilePath,'wb') as f:
                    f.write(response.content)
            except:
                return{'msg':'Error','description':'Unable to save downloaded file.'}
            ################ Read Image from Base64 string ################################
            try:
                CurrentImage=cv2.imread(DownloadFilePath)
                GrayImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
                #cv2.imwrite("Test.jpg", CurrentImage)
                #os.remove(DownloadFilePath)
            except:
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to read downladed image.'}
            #################### Perform OCR using Tesseract #####################################
            try:
                ConvertedImageDF=PerformPassportFrontOCRTesseract(CurrentImage)
                ConvertedImageDF=CleanPassportFrontData(ConvertedImageDF)
                print(list(ConvertedImageDF['Word']))
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Corrupted Image - Unable to Perform OCR'}
            try:
                ################## Check if Tesseract returned all valid keywords #######################
                TesseractWorks=False
                DatesValid=False
                PassportNumberValid=False
                PlaceValid=False
                SurnameValid=False
                GivenNameValid=False
                NationalityValid=False
                FirstPlaceTop=""
                ############# Validity Check For Dates and Passport ################
                Dates=[]
                for word in list(ConvertedImageDF['Word']):
                    if re.match(r"^[0-9]{2}\/[0-9]{2}\/[0-9]{4}$",word.strip()):
                        Dates.append(word.strip())
                    elif re.match(r"^[A-Z]{1}[0-9]{7}$",word.strip()):
                        PassportNumber = word.strip()
                DatesValid = len(Dates) == 3
                print("Dates Valid: ",DatesValid)
                print("")
                PassportNumberValid = PassportNumber!=""
                print("Passport Number Valid: ",PassportNumberValid)
                print("")
                ############# Validity Check For Place ################
                PlaceDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.startswith('PLA')) | (ConvertedImageDF['Word'].str.endswith('ACE'))].reset_index(drop=True)
                PlaceValid = PlaceDF.shape[0]==2
                print("Place Valid: ",PlaceValid)
                print("")
                ############# Validity Check For Surname ################
                SurnameDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.endswith("URNAME")) | (ConvertedImageDF['Word'].str.startswith("SURN"))]
                SurnameValid = SurnameDF.shape[0] == 1
                print("Surname Valid: ",SurnameValid)
                print("")
                ############# Validity Check For Given Name ################
                GivenDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.endswith("VEN")) | (ConvertedImageDF['Word'].str.startswith("GIV"))]
                GivenNameValid = GivenDF.shape[0] == 1
                print("Given Name Valid: ",GivenNameValid)
                print("")
                ############# Validity Check For Nationality ################
                NationalityDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.startswith("NATION")) | (ConvertedImageDF['Word'].str.endswith("NALITY"))]
                NationalityValid = NationalityDF.shape[0] == 1
                print("Nationality Valid: ",NationalityValid)
                print("")
                ############ Validity Check for whole dataframe
                TesseractWorks = DatesValid and PassportNumberValid and PlaceValid and SurnameValid and GivenNameValid and NationalityValid
            except Exception as e:
                print(e)
                os.remove(DownloadFilePath)
                return {'Msg':'Error','Description':'Unable to check if all required keywords are present or not in Tesseract output'}
            ########################## Proceed with tesseract output if Tesseract works ####################################
            if TesseractWorks:
                ################ Get Date Of Birth, Issue and Expiry ##################################
                try:
                    DatesDF = ConvertedImageDF[ConvertedImageDF['Word'].isin(Dates)]
                    PlaceBottom = PlaceDF['Bottom'][0]
                    DOBDF = DatesDF[DatesDF['Bottom']<PlaceBottom].reset_index(drop=True)
                    OtherDates = DatesDF[DatesDF['Top']>PlaceBottom].reset_index(drop=True)
                    if DOBDF.shape[0] == 1:
                        DateOfBirth = DOBDF['Word'][0]
                    if OtherDates.shape[0] == 2:
                        OtherDates = OtherDates.sort_values(by='Left').reset_index(drop=True)
                        DateOfIssue = OtherDates['Word'][0].strip()
                        DateOfExpiry = OtherDates['Word'][1].strip()
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get dates from Tesseract output'}
                ####################### Get Sex #########################################
                try:
                    for word in list(ConvertedImageDF['Word']):
                        if word.strip() == "F":
                            Sex = "Female"
                            break
                        elif word.strip() == "M":
                            Sex = "Male"
                            break
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get Sex from Tesseract output'}
                ################ Get Surname and Given Names ################################
                try:
                    SurnameBottom = list(SurnameDF['Bottom'])[0]
                    GivenTop = list(GivenDF['Top'])[0]
                    GivenBottom = list(GivenDF['Bottom'])[0]
                    NationalityTop = list(NationalityDF['Top'])[0]
                    SurnameValueDF = ConvertedImageDF[(ConvertedImageDF['Top'] > SurnameBottom) & (ConvertedImageDF['Bottom'] < GivenTop+10)]
                    SurnameValueDF=SurnameValueDF.sort_values(by='Left').reset_index(drop=True)
                    Surname = GetValidValues(SurnameValueDF)
                    GivenNameDF = ConvertedImageDF[(ConvertedImageDF['Top'] > GivenBottom) & (ConvertedImageDF['Bottom'] < NationalityTop+10)]
                    GivenNameDF=GivenNameDF.sort_values(by='Left').reset_index(drop=True)
                    GivenName = GetValidValues(GivenNameDF)
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get Surname and Given Name from Tesseract output'}
                ##################### Get Place Of Issue and Place Of Birth #####################################
                try:
                    PlaceDFSorted = PlaceDF.sort_values(by='Top').reset_index(drop=True)
                    LastPlaceBottom = PlaceDFSorted['Bottom'][1]
                    ConvertedImageDFAfterPlaceOfIssue = ConvertedImageDF[ConvertedImageDF['Bottom'] > LastPlaceBottom]
                    ConvertedImageDFAfterPlaceOfIssue = ConvertedImageDFAfterPlaceOfIssue.sort_values(by='Top')
                    AfterPlaceOfIssueWords=list(ConvertedImageDFAfterPlaceOfIssue['Word'])
                    for word in AfterPlaceOfIssueWords:
                        for place in ListOfPlaceOfIssue:
                            if place in word.strip():
                                PlaceOfIssue = place
                                break
                    FirstPlaceBottom = PlaceDFSorted['Bottom'][0]
                    FirstPlaceTop = PlaceDFSorted['Top'][0]
                    PlaceOfBirthDF = ConvertedImageDF[(ConvertedImageDF['Bottom'] > FirstPlaceBottom+10) &
                                                      (ConvertedImageDF['Bottom'] < LastPlaceBottom-10)]
                    PlaceOfBirthDF = PlaceOfBirthDF.sort_values(by='Left').reset_index(drop=True)
                    PlaceOfBirth = GetValidValues(PlaceOfBirthDF)
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get Place Of Issue and Place Of Birth from Tesseract output'}
                ########################## Get Nationality #########################################
                try:
                    if (FirstPlaceTop != ""):
                        NationalityBottom = list(NationalityDF['Bottom'])[0]
                        NationalityValuesDF = ConvertedImageDF[(ConvertedImageDF['Bottom']>NationalityBottom) &
                        (ConvertedImageDF['Bottom']<FirstPlaceTop)]
                        print(NationalityValuesDF)
                        NationalityValuesDF = NationalityValuesDF.sort_values(by='Left').reset_index(drop=True)
                        Nationality = GetValidValues(NationalityValuesDF).split()[0]
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get Nationality from Tesseract output'}
                ###################### Return Response ################################
                ResponseDict=dict(Msg='Success',PassportNumber=PassportNumber,Surname=Surname,GivenName=GivenName,
                Nationality=Nationality,Sex=Sex,DateOfBirth=DateOfBirth,PlaceOfIssue=PlaceOfIssue,
                DateOfIssue=DateOfIssue,DateOfExpiry=DateOfExpiry,Method="Tesseract")
                return ResponseDict
            else:
                print("Unable to Fetch Values using Tesseract so proceeeding for Google Vision API !!")
                print("")
                ################ Get Dataframe from Google Vision API ######################
                try:
                    CurrentImage=cv2.imread(DownloadFilePath)
                    GrayImage=cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(DownloadFilePath,GrayImage)
                    WordsAndCoordinatesDF=PerformOCRGoogleVisionAPI(DownloadFilePath)
                    os.remove(DownloadFilePath)
                    ################ Check Response from Google Vision API ######################
                    if str(type(WordsAndCoordinatesDF)) != "<class 'pandas.core.frame.DataFrame'>":
                        return {'Msg':'Error','Description':'Unable to Perform OCR using Google Vision API - Poor Image Quality.'}
                except Exception as e:
                    print(e)
                    if os.path.exists(DownloadFilePath):
                        os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to perform OCR using Google Vision API.'}
                ################ Get top, bottom, left, right from Google Vision API ######################
                try:
                    ConvertedImageDF = WordsAndCoordinatesDF.copy()
                    ConvertedImageDF['Top']=ConvertedImageDF.apply(func=CreateTop,axis=1)
                    ConvertedImageDF['Bottom']=ConvertedImageDF.apply(func=CreateBottom,axis=1)
                    ConvertedImageDF['Left']=ConvertedImageDF.apply(func=CreateLeft,axis=1)
                    ConvertedImageDF['Right']=ConvertedImageDF.apply(func=CreateRight,axis=1)
                    ConvertedImageDF=ConvertedImageDF[['Word','Top','Bottom','Left','Right']]
                    print(list(ConvertedImageDF['Word']))
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to create top/bottom/left/right from Google Vision API response.'}
                ################ Clean Passport Front Page Data #############################
                try:
                    ConvertedImageDF = CleanPassportFrontData(ConvertedImageDF)
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to clean Google Vision API response.'}
                ################## Check if Google Vision API returned all valid keywords #######################
                try:
                    GVAWorks=False
                    DatesValid=False
                    PassportNumberValid=False
                    PlaceValid=False
                    SurnameValid=False
                    GivenNameValid=False
                    NationalityValid=False
                    FirstPlaceTop=""
                    ############# Validity Check For Dates and Passport ################
                    Dates=[]
                    for word in list(ConvertedImageDF['Word']):
                        if re.match(r"^[0-9]{2}\/[0-9]{2}\/[0-9]{4}$",word.strip()):
                            Dates.append(word.strip())
                        elif re.match(r"^[A-Z]{1}[0-9]{7}$",word.strip()):
                            PassportNumber = word.strip()
                    DatesValid = len(Dates) == 3
                    print("Dates Valid: ",DatesValid)
                    print("")
                    PassportNumberValid = PassportNumber!=""
                    print("Passport Number Valid: ",PassportNumberValid)
                    print("")
                    ############# Validity Check For Place ################
                    PlaceDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.startswith('PLA')) | (ConvertedImageDF['Word'].str.endswith('ACE'))].reset_index(drop=True)
                    PlaceValid = PlaceDF.shape[0]==2
                    print("Place Valid: ",PlaceValid)
                    print("")
                    ############# Validity Check For Surname ################
                    SurnameDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.endswith("URNAME")) | (ConvertedImageDF['Word'].str.startswith("SURN"))]
                    SurnameValid = SurnameDF.shape[0] == 1
                    print("Surname Valid: ",SurnameValid)
                    print("")
                    ############# Validity Check For Given Name ################
                    GivenDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.endswith("VEN")) | (ConvertedImageDF['Word'].str.startswith("GIV"))]
                    GivenNameValid = GivenDF.shape[0] == 1
                    print("Given Name Valid: ",GivenNameValid)
                    print("")
                    ############# Validity Check For Nationality ################
                    NationalityDF = ConvertedImageDF[(ConvertedImageDF['Word'].str.startswith("NATION")) | (ConvertedImageDF['Word'].str.endswith("NALITY"))]
                    NationalityValid = NationalityDF.shape[0] == 1
                    print("Nationality Valid: ",NationalityValid)
                    print("")
                    ############ Validity Check for whole dataframe
                    GVAWorks = DatesValid and PassportNumberValid and PlaceValid and SurnameValid and GivenNameValid and NationalityValid
                    if not(GVAWorks):
                        return {'Msg':'Error','Description':'All required keywords are not present in GVA output'}
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to check if all required keywords are present or not in GVA output'}
                ################ Get Date Of Birth, Issue and Expiry ##################################
                try:
                    DatesDF = ConvertedImageDF[ConvertedImageDF['Word'].isin(Dates)]
                    PlaceBottom = PlaceDF['Bottom'][0]
                    DOBDF = DatesDF[DatesDF['Bottom']<PlaceBottom].reset_index(drop=True)
                    OtherDates = DatesDF[DatesDF['Top']>PlaceBottom].reset_index(drop=True)
                    if DOBDF.shape[0] == 1:
                        DateOfBirth = DOBDF['Word'][0]
                    if OtherDates.shape[0] == 2:
                        OtherDates = OtherDates.sort_values(by='Left').reset_index(drop=True)
                        DateOfIssue = OtherDates['Word'][0].strip()
                        DateOfExpiry = OtherDates['Word'][1].strip()
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to get dates from GVA output'}
                ################# Get Sex and Passport Number ############################
                try:
                    for word in list(ConvertedImageDF['Word']):
                        if word.strip() == "F":
                            Sex = "Female"
                            break
                        elif word.strip() == "M":
                            Sex = "Male"
                            break
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to get Sex from GVA output'}
                ################ Get Surname and Given Names ################################
                try:
                    SurnameBottom = list(SurnameDF['Bottom'])[0]
                    GivenTop = list(GivenDF['Top'])[0]
                    GivenBottom = list(GivenDF['Bottom'])[0]
                    NationalityTop = list(NationalityDF['Top'])[0]
                    SurnameValueDF = ConvertedImageDF[(ConvertedImageDF['Bottom'] > SurnameBottom) & (ConvertedImageDF['Top'] < GivenTop-15)]
                    SurnameValueDF=SurnameValueDF.sort_values(by='Left').reset_index(drop=True)
                    Surname = GetValidValues(SurnameValueDF)
                    GivenNameDF = ConvertedImageDF[(ConvertedImageDF['Bottom'] > GivenBottom) & (ConvertedImageDF['Top'] < NationalityTop-20)]
                    GivenNameDF=GivenNameDF.sort_values(by='Left').reset_index(drop=True)
                    GivenName = GetValidValues(GivenNameDF)
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to get Surname and Given Name from GVA output'}
                ##################### Get Place Of Issue and Place Of Birth #####################################
                try:
                    PlaceDFSorted = PlaceDF.sort_values(by='Top').reset_index(drop=True)
                    LastPlaceBottom = PlaceDFSorted['Bottom'][1]
                    ConvertedImageDFAfterPlaceOfIssue = ConvertedImageDF[ConvertedImageDF['Bottom'] > LastPlaceBottom]
                    ConvertedImageDFAfterPlaceOfIssue = ConvertedImageDFAfterPlaceOfIssue.sort_values(by='Top')
                    AfterPlaceOfIssueWords=list(ConvertedImageDFAfterPlaceOfIssue['Word'])
                    for word in AfterPlaceOfIssueWords:
                        for place in ListOfPlaceOfIssue:
                            if place in word.strip():
                                PlaceOfIssue = place
                                break
                    FirstPlaceBottom = PlaceDFSorted['Bottom'][0]
                    FirstPlaceTop = PlaceDFSorted['Top'][0]
                    PlaceOfBirthDF = ConvertedImageDF[(ConvertedImageDF['Bottom'] > FirstPlaceBottom+10) &
                                                      (ConvertedImageDF['Bottom'] < LastPlaceBottom-10)]
                    PlaceOfBirthDF = PlaceOfBirthDF.sort_values(by='Left').reset_index(drop=True)
                    PlaceOfBirth = GetValidValues(PlaceOfBirthDF)
                except Exception as e:
                    print(e)
                    os.remove(DownloadFilePath)
                    return {'Msg':'Error','Description':'Unable to get Place Of Issue and Place Of Birth from GVA output'}
                ########################## Grab Nationality #########################################
                try:
                    if (FirstPlaceTop != ""):
                        NationalityBottom = list(NationalityDF['Bottom'])[0]
                        NationalityValuesDF = ConvertedImageDF[(ConvertedImageDF['Bottom']>NationalityBottom) &
                        (ConvertedImageDF['Top']<FirstPlaceTop-15)]
                        NationalityValuesDF = NationalityValuesDF.sort_values(by='Left').reset_index(drop=True)
                        Nationality = GetValidValues(NationalityValuesDF).split()[0]
                except Exception as e:
                    print(e)
                    return {'Msg':'Error','Description':'Unable to get Nationality from GVA output'}
                ###################### Return Response ################################
                ResponseDict=dict(Msg='Success',PassportNumber=PassportNumber,Surname=Surname,GivenName=GivenName,
                Nationality=Nationality,Sex=Sex,DateOfBirth=DateOfBirth,PlaceOfIssue=PlaceOfIssue,
                DateOfIssue=DateOfIssue,DateOfExpiry=DateOfExpiry,Method="GoogleVisionAPI")
                return ResponseDict
        except Exception as e:
            print(e)
            if os.path.exists(DownloadFilePath):
                os.remove(DownloadFilePath)
            return {'Msg':'Error','Description':'Unknown Exception Happened. Please make sure that the Image Orientation is upright.'}
#################### Configure URLs #########################
api.add_resource(PANCardOCR,'/PancardOCR')
api.add_resource(AadharFrontOCR,'/AadharFrontOCR')
api.add_resource(AadharBackOCR,'/AadharBackOCR')
api.add_resource(PassportFrontOCR,'/PassportFrontOCR')
#################  Run Flask Server ##########################
if __name__ == '__main__':
    app.run(debug = True,host='0.0.0.0')
