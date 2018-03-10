import cv2
import numpy as np

print ("import library")
class Harris:
    def __init__(self,imgname):
        self.imgname = imgname
    def readGreyM(self):
        img = cv2.imread(self.imgname)
        imgGrey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        print("read grey img")
        self.img = img
        #print("Lena image intensity array\n:%s"%(lenaGrey))
        #transfer lena's image intensity into a numpy matrix
        
        self.mGrey = np.matrix(imgGrey)
        self.HEIGHT = int(self.mGrey.shape[0])
        self.WIDTH = int(self.mGrey.shape[1])
        
        #print image attributes
        print(self.mGrey)
        print("Width:%s,Height:%s"%(self.WIDTH,self.HEIGHT))
    
    def checkBorder(self,va,borderA,vb,borderB):
        if va-1>=0 and va+1<borderA and vb-1>=0 and vb+1<borderB:
            return True
        else:
            return False

    def calGradient(self):
        self.iSharp = self.mGrey
        GradientX = self.mGrey
        GradientY = self.mGrey
        # i for y axis HEIGHT
        # j for x axis WIDTH
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if self.checkBorder(i,self.HEIGHT,j,self.WIDTH):
                    #calculate gray scale gradient in x and y axes
                    dx = abs(int(self.mGrey[i,j+1])-int(self.mGrey[i,j]))

                    dy = abs(int(self.mGrey[i+1,j])-int(self.mGrey[i,j]))

                    GradientX[i,j] = dx
                    GradientY[i,j] = dy 
                    self.iSharp[i,j] = max(dx,dy)
                else:
                    GradientX[i,j] = 0
                    GradientY[i,j] = 0
                    self.iSharp[i,j] = 0
                
                
        self.Ix = np.array(GradientX)
        self.Iy = np.array(GradientY)

        print("X Gradient:\n%s"%(self.Ix))
        print("Y Gradient:\n%s"%(self.Iy))
    
    #paper method
    def showSharp(self):
        cv2.imshow("sharp",self.iSharp)
        cv2.waitKey(0)
    def calGradientM(self):
        self.Ix,self.Iy=np.gradient(self.mGrey)
        print("X Gradient:\n%s"%(self.Ix))
        print("Y Gradient:\n%s"%(self.Iy))
        
    #use GaussianBlur to process parameter A,B,C  
    def blurPara(self):
        self.A = cv2.GaussianBlur(self.Ix*self.Ix,(3,3),1.5)
        self.B = cv2.GaussianBlur(self.Iy*self.Iy,(3,3),1.5)
        self.C = cv2.GaussianBlur(self.Ix*self.Iy,(3,3),1.5)
        
        print(self.A)
        print(self.B)
        print(self.C)

    def calR(self):
        #R = detM-k(traceM)^2
        
        R = np.zeros(self.mGrey.shape)
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                M = [[self.A[i,j],self.C[i,j]],[self.C[i,j],self.B[i,j]]]

                R[i,j] = np.linalg.det(M)-0.06*(np.trace(M))*(np.trace(M))
                #R[i,j] = self.A[i,j]*self.B[i,j]-pow(self.C[i,j],2)-0.06*pow(self.A[i,j]+self.B[i,j],2)
        self.R = R
        print(self.R)

    
    def processR(self,threshold):
        pR = self.R
        R = self.R
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                #threshold
                if R[i,j]<np.amin(R)*threshold:
                    pR[i,j] = 255
                    #self.img[i,j] = [255,0,0]
                    #blue green red
                    cv2.circle(self.img,(j,i),5,(203,192,255),0)
        self.pR = pR

    def display(self):                    
        cv2.namedWindow('R',cv2.WINDOW_NORMAL)
        cv2.imshow('R',self.R)

        cv2.waitKey(0)
        cv2.imshow('R',self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


H = Harris("taiku.jpg")
H.readGreyM()
H.calGradient()
#H.showSharp()
H.blurPara()
H.calR()
H.processR(0.8)
H.display()