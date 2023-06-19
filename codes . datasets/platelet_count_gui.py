import tkinter as Tk 
from tkinter import *
import cv2
import numpy as np
from time import sleep
import scipy.ndimage as ndi
import glob
import smtplib

'''
*Author List :  Akshatha K V,Divya A Jamakhandi, Disha B Hegde,
*File Name : platelet_count_gui.py
*Functions : buttonClickExecute() -Button event in which image Processing is done
*Global Variables : ---
*Procedure of executing: Run the platelet_count_gui.py file
*Algorithm:
     Convert image into grayscale
     Use OTSU thresholding to convert it into black and white
     Use contour plots to cover up all the platelets
     Extract only the platelets from the original black&white image (find the difference between the two images)
     invert the black and white image to obtain white platelets on a black background
     Find clusters of 1's in the black&white image to obtain the number of platelets in the given image
     Multiply the number obtained by a scalar to obtain a projected count of the number of platelets
'''

#GUI for the Platelet Counter
master = Tk()
master.geometry("255x200")
master.title("Platelet Counter")

options=["1","2","3","4","5"]

variable = StringVar(master)
variable.set('select an image number')

master.title("Platelet Counter")

label1 = Label(master,text="   ")
label1.grid(row=0,column=0)

label1 = Label(master,text="   ")
label1.grid(row=1,column=0)
labelname = Label(master,text="NAME")
labelname.grid(row=1,column=1)
entryname = Entry(master)
entryname.grid(row=1,column=2)

label1 = Label(master,text="   ")
label1.grid(row=2,column=1)

label1 = Label(master,text="   ")
label1.grid(row=3,column=0)
labelmail = Label(master,text="MAIL_ID")
labelmail.grid(row=3,column=1)
entrymail = Entry(master)
entrymail.grid(row=3,column=2)

label1 = Label(master,text="   ")
label1.grid(row=4,column=1)

w = OptionMenu(master,variable, *options)
w.grid(row=5,column=2)

label1 = Label(master,text="   ")
label1.grid(row=6,column=2)

#event on botton press
def buttonClickExecute():
     print(" Hello " + entryname.get()+ "\n Mail_ID: " + entrymail.get()  + "\n Image selected: Platelet_image_" + variable.get()  )
     count_img=0

     #image import
     for imag in glob.glob("E:\\Platelet\\platelet"+ variable.get() + ".jpg"):

             #image read
             img=cv2.imread(imag)
             img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
             
             #OTSU Thresholding
             ret,thr =cv2.threshold(img,0,255,cv2.THRESH_OTSU)
             m,n=thr.shape
             thr2=thr.copy()

             #filling holes
             hello,contours, hier = cv2.findContours(thr,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

             count=0;
             for cnt in contours:
                     if cv2.contourArea(cnt)<80 and cv2.contourArea(cnt)>20:
                             count=count+1
                             cv2.drawContours(img,[cnt],0,(0,255,0),2)
                             cv2.drawContours(thr,[cnt],0,255,-1)

             #creating black mask            
             plate = np.zeros(shape=(m,n))

             #after thresholding image platelets in white on black backround
             for i in range(0,m):
                     for j in range(0,n):
                             if thr2[i][j]==thr[i][j]:
                                     plate[i][j]=255
                             else:
                                     plate[i][j]=thr2[i][j]
                                     
             #inverting black and white regions                  
             for i in range(0,m):
                     for j in range(0,n):
                             if plate[i][j]==0:
                                     plate[i][j]=255
                             else:
                                     plate[i][j]=0
                                     
             #counting the pletelets(black clusters)
             labeled_array, num_features = ndi.label(plate)
             count_img=count_img+1
             
             print(" Platelet count: " + str(num_features))

             #emailing the results to the entered mail id
             content= ''+''+'Hello ' + entryname.get()+ "\n Greetings from Platelet project RVCE. \n Thankyou for participating in this test run of Platelet counter. \n The details of selected image and count is given below:" + "\n \n Image selected: Platelet_image_" + variable.get() + "\n Platelet count: " + str(num_features)
             mail=smtplib.SMTP('smtp.gmail.com',587)
             mail.ehlo()
             mail.starttls()
             mail.login('plateletprojectrvce@gmail.com','plateletproject2018')
             mail.sendmail('plateletprojectrvce@gmail.com',str(entrymail.get()),content)
             mail.close()           
             
     #displaying image of which the count is being displayed
     display = cv2.imread("E:\\Platelet\\platelet"+ variable.get() + ".jpg")
     cv2.imshow('Image',display)
     cv2.waitKey(3000)  

button = Button(master,text="OK", command=buttonClickExecute)
button.grid(row=7,column=2)

master.mainloop()
