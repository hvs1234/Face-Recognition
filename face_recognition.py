from tkinter import *
from tkinter import messagebox
root = Tk()
root.geometry('800x500+300+60')
root.resizable(False,False)
root.title('Login ...')
root.configure(bg='orange')
# ------------------------------------------------------------------------------------------------------
def f1():
    a = e1.get();  b = e2.get();
    if(a=="" and b==""):
        messagebox.showwarning('Blanked Login ...','BLanked Not Allowed !')

    elif(a=="harsh" and b=="h1234"):
        messagebox.showinfo('Login','Login Success')
        root.destroy();
        import numpy as np
        import cv2 as cv

        cascade = cv.CascadeClassifier('haar_face.xml')
        people = ['tim_cook','satya_nadella','sundar_pichai','chris_evans','jeff_bezos','tony_stark',
        'hritik_roshan']
        face_recognize = cv.face.LBPHFaceRecognizer_create()
        face_recognize.read('face_trained.yml')

        image = cv.imread(r'./Resources/Faces/val/jeff_bezos/2.jpeg')

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces_rect = cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h,x:x+w]

            label, confidence = face_recognize.predict(faces_roi)
            #print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.putText(image, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=2)

        cv.imshow('Face Detect', image)

        cv.waitKey(0)
    else:
        messagebox.showerror('Invalid Login','Invalid username and Password')

def f2():
    root2 = Tk()
    root2.geometry('600x200+600+100')
    root2.title('...');  root2.resizable(False,False)
    root2.configure(bg='sky blue');
    l1 = Label(root2,text='NOTE:- Every time during login, it is neccessary to enter\n username or password.'
            ' Your login portal will reset every\n time when you put a new value during face recognition.'
            ' But remember\n one thing, blanked username or password is not allowed.\n When you will do login'
            ' successfully then click "Keep me logged in ..."',fg='indigo',bg='sky blue',
            font = 'Cambria 10 bold').place(x=20,y=30)
    root2.mainloop()
# ------------------------------------------------------------------------------------------------------
l1 = Label(root,text = 'L-O-G-I-N ...',bg='orange',fg='black',font='Cambria 30 bold').place(x=140,y=60)
l2 = Label(root,text = 'Username ',bg='orange',fg='black',font='Cambria 15 bold').place(x=140,y=170)
l3 = Label(root,text = 'Password ',bg='orange',fg='black',font='Cambria 15 bold').place(x=140,y=260)
# -------------------------------------------------------------------------------------------------------
s1 = StringVar()
e1 = Entry(root,textvariable = s1,
        bd=4,fg='indigo',bg='light gray',font='Cambria 10 bold',width=40); e1.place(x=280,y=170)
s1.set("harsh")
s2 = StringVar()
e2 = Entry(root,textvariable = s2,
        bd=4,fg='indigo',bg='light gray',font='Cambria 10 bold',width=40,
            show="*"); e2.place(x=280,y=260)
s2.set("h1234")
# -------------------------------------------------------------------------------------------------------
chk1 = Radiobutton(root,text = ' Keep me logged in ...',font='Cambria 12 bold',bg='orange',fg='black'
        ).place(x=140,y=340)
# -------------------------------------------------------------------------------------------------------
b1 = Button(root,text = 'Login',width=6,bg='orange',fg='black', command = f1,
            font='Cabria 15 bold',bd=4); b1.place(x=140,y=410)
b2 = Button(root,text = 'Cancel',width=6,bg='orange',fg='black',font='Cabria 15 bold',bd=4,
            command = root.destroy); b2.place(x=300,y=410)
b3 = Button(root,text='...',bd=4,bg='orange',fg='indigo',width=1,command=f2).place(x=700,y=410)
root.mainloop()
