###This file contains the utility functions that are commonly used in EDER_VIANN packages

import csv
import math
import os, shutil
import subprocess
#import time as t
import datetime
import ast
import numpy as np
#import scipy
import copy
#from scipy.stats import chisquare

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"

class Track:
      def __init__(self,segments):
          self.SegmentHeader=sorted(segments, key=str.lower)
          self.Segmentation=len(self.SegmentHeader)
      def __eq__(self, other):
        return ('-'.join(self.SegmentHeader)) == ('-'.join(other.SegmentHeader))
      def __hash__(self):
        return hash(('-'.join(self.SegmentHeader)))
      def DecorateSegments(self,RawHits): #Decorate hit information
          self.SegmentHits=[]
          for s in range(len(self.SegmentHeader)):
              self.SegmentHits.append([])
              for t in RawHits:
                   if self.SegmentHeader[s]==t[3]:
                      self.SegmentHits[s].append(t[:3])
          for Hit in range(0, len(self.SegmentHits)):
             self.SegmentHits[Hit]=sorted(self.SegmentHits[Hit],key=lambda x: float(x[2]),reverse=False)

      def DecorateTrackGeoInfo(self):
          if hasattr(self,'SegmentHits'):
             if self.Segmentation==2:
                __XZ1=Track.GetEquationOfTrack(self.SegmentHits[0])[0]
                __XZ2=Track.GetEquationOfTrack(self.SegmentHits[1])[0]
                __YZ1=Track.GetEquationOfTrack(self.SegmentHits[0])[1]
                __YZ2=Track.GetEquationOfTrack(self.SegmentHits[1])[1]
                __X1S=Track.GetEquationOfTrack(self.SegmentHits[0])[3]
                __X2S=Track.GetEquationOfTrack(self.SegmentHits[1])[3]
                __Y1S=Track.GetEquationOfTrack(self.SegmentHits[0])[4]
                __Y2S=Track.GetEquationOfTrack(self.SegmentHits[1])[4]
                __Z1S=Track.GetEquationOfTrack(self.SegmentHits[0])[5]
                __Z2S=Track.GetEquationOfTrack(self.SegmentHits[1])[5]
                __vector_1_st = np.array([np.polyval(__XZ1,self.SegmentHits[0][0][2]),np.polyval(__YZ1,self.SegmentHits[0][0][2]),self.SegmentHits[0][0][2]])
                __vector_1_end = np.array([np.polyval(__XZ1,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),np.polyval(__YZ1,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),self.SegmentHits[0][len(self.SegmentHits[0])-1][2]])
                __vector_2_st = np.array([np.polyval(__XZ2,self.SegmentHits[0][0][2]),np.polyval(__YZ2,self.SegmentHits[0][0][2]),self.SegmentHits[0][0][2]])
                __vector_2_end = np.array([np.polyval(__XZ2,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),np.polyval(__YZ2,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),self.SegmentHits[0][len(self.SegmentHits[0])-1][2]])
                __result=Track.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
                __midpoint=(__result[0]+__result[1])/2
                __v1=np.subtract(__vector_1_end,__midpoint)
                __v2=np.subtract(__vector_2_end,__midpoint)
                if self.SegmentHits[0][len(self.SegmentHits)-1][2]>self.SegmentHits[1][len(self.SegmentHits)-1][2]: #Workout which track is leading (has highest z-coordinate)
                    __leading_seg=0
                    __subleading_seg=1
                else:
                    __leading_seg=1
                    __subleading_seg=0
                self.angle=Track.angle_between(__v1, __v2)
                self.DOCA=__result[2]
                self.Seg_Lon_Gap=float(self.SegmentHits[__leading_seg][0][2])-float(self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][2])
                __x2=float(self.SegmentHits[__leading_seg][0][0])
                __x1=self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][0]
                __y2=float(self.SegmentHits[__leading_seg][0][1])
                __y1=self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][1]
                self.Seg_Transv_Gap=math.sqrt(((__x2-__x1)**2)+((__y2-__y1)**2))
             else:
                 raise ValueError("Method 'DecorateTrackGeoInfo' currently works for track segment combinations with number of segments of 2 only")
          else:
                raise ValueError("Method 'DecorateTrackGeoInfo' works only if 'DecorateTracks' method has been acted upon the seed before")

      def TrackQualityCheck(self,MaxDoca,MaxSLG, MaxSTG,MaxAngle):
                    self.GeoFit = (self.DOCA<=MaxDoca and self.Seg_Lon_Gap<=MaxSLG and self.Seg_Transv_Gap<=(MaxSTG*(self.Seg_Lon_Gap*0.96)) and abs(self.angle)<=MaxAngle)

      def CNNFitTrack(self,Prediction):
          self.Track_CNN_Fit=Prediction

      def AssignCNNVxId(self,ID):
          self.VX_CNN_ID=ID

      def InjectTrack(self,OtherTrack):
          __overlap=False
          for t1 in self.SegmentHeader:
              for t2 in OtherTrack.SegmentHeader:
                  if t1==t2:
                      __overlap=True
                      break
          if __overlap:
              overlap_matrix=[]
              for t1 in range(len(self.SegmentHeader)):
                 for t2 in range(len(OtherTrack.SegmentHeader)):
                    if self.SegmentHeader[t1]==OtherTrack.SegmentHeader[t2]:
                       overlap_matrix.append(t2)

              for t2 in range(len(OtherTrack.SegmentHeader)):
                if (t2 in overlap_matrix)==False:
                  self.SegmentHeader.append(OtherTrack.SegmentHeader[t2])
                  if hasattr(self,'SegmentHits') and hasattr(OtherTrack,'SegmentHits'):
                          self.SegmentHits.append(OtherTrack.SegmentHits[t2])
              if hasattr(self,'MC_truth_label') and hasattr(OtherTrack,'MC_truth_label'):
                         self.MC_truth_label=(self.MC_truth_label and OtherTrack.MC_truth_label)
              if hasattr(self,'VX_CNN_Fit') and hasattr(OtherTrack,'VX_CNN_Fit'):
                        self.VX_CNN_Fit+=OtherTrack.VX_CNN_Fit
              elif hasattr(self,'VX_CNN_Fit'):
                       self.VX_CNN_Fit.append(OtherTrack.Track_CNN_Fit)
              elif hasattr(OtherTrack,'VX_CNN_Fit'):
                       self.VX_CNN_Fit=[self.Track_CNN_Fit]
                       self.VX_CNN_Fit+=OtherTrack.VX_CNN_Fit
              else:
                      self.VX_CNN_Fit=[]
                      self.VX_CNN_Fit.append(self.Track_CNN_Fit)
                      self.VX_CNN_Fit.append(OtherTrack.Track_CNN_Fit)
              if hasattr(self,'VX_x') and hasattr(OtherTrack,'VX_x'):
                      self.VX_x+=OtherTrack.VX_x
              elif hasattr(self,'VX_x'):
                       self.VX_x.append(OtherTrack.Vx)
              elif hasattr(OtherTrack,'VX_x'):
                       self.VX_x=[self.Vx]
                       self.VX_x+=OtherTrack.VX_x
              else:
                      self.VX_x=[]
                      self.VX_x.append(self.Vx)
                      self.VX_x.append(OtherTrack.Vx)
                          
              if hasattr(self,'VX_y') and hasattr(OtherTrack,'VX_y'):
                      self.VX_y+=OtherTrack.VX_y
              elif hasattr(self,'VX_y'):
                       self.VX_y.append(OtherTrack.Vy)
              elif hasattr(OtherTrack,'VX_y'):
                       self.VX_y=[self.Vy]
                       self.VX_y+=OtherTrack.VX_y
              else:
                      self.VX_y=[]
                      self.VX_y.append(self.Vy)
                      self.VX_y.append(OtherTrack.Vy)
                  
              if hasattr(self,'VX_z') and hasattr(OtherTrack,'VX_z'):
                      self.VX_z+=OtherTrack.VX_z
              elif hasattr(self,'VX_z'):
                       self.VX_z.append(OtherTrack.Vz)
              elif hasattr(OtherTrack,'VX_z'):
                       self.VX_z=[self.Vz]
                       self.VX_z+=OtherTrack.VX_z
              else:
                      self.VX_z=[]
                      self.VX_z.append(self.Vz)
                      self.VX_z.append(OtherTrack.Vz)
              self.VX_z=list(set(self.VX_z))
              self.VX_x=list(set(self.VX_x))
              self.VX_y=list(set(self.VX_y))
              self.VX_CNN_Fit=list(set(self.VX_CNN_Fit))
              self.Segmentation=len(self.SegmentHeader)
              self.Track_CNN_Fit=sum(self.VX_CNN_Fit)/len(self.VX_CNN_Fit)
              self.Vx=sum(self.VX_x)/len(self.VX_x)
              self.Vy=sum(self.VX_y)/len(self.VX_y)
              self.Vz=sum(self.VX_z)/len(self.VX_z)
              if hasattr(self,'angle'):
                  delattr(self,'angle')
              if hasattr(self,'DOCA'):
                  delattr(self,'DOCA')
              if hasattr(self,'V_Tr'):
                  delattr(self,'V_Tr')
              if hasattr(self,'Tr_Tr'):
                  delattr(self,'Tr_Tr')



              return True

          else:
              return __overlap


      def MCtruthClassifyTrack(self,label):
          self.MC_truth_label=label

      def PrepareTrackPrint(self,MaxX,MaxY,MaxZ,Res,Rescale):
          __TempTrack=copy.deepcopy(self.SegmentHits)
          self.Resolution=Res
          self.bX=int(round(MaxX/self.Resolution,0))
          self.bY=int(round(MaxY/self.Resolution,0))
          self.bZ=int(round(MaxZ/self.Resolution,0))
          self.H=(self.bX)*2
          self.W=(self.bY)*2
          self.L=(self.bZ)
          __LongestDistance=0.0
          for __Track in __TempTrack:
            __Xdiff=float(__Track[len(__Track)-1][0])-float(__Track[0][0])
            __Ydiff=float(__Track[len(__Track)-1][1])-float(__Track[0][1])
            __Zdiff=float(__Track[len(__Track)-1][2])-float(__Track[0][2])
            __Distance=math.sqrt(__Xdiff**2+__Ydiff**2+__Zdiff**2)
            if __Distance>=__LongestDistance:
                __LongestDistance=__Distance
                __FinX=float(__Track[0][0])
                __FinY=float(__Track[0][1])
                __FinZ=float(__Track[0][2])
                self.LongestTrackInd=__TempTrack.index(__Track)
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=float(__Hits[0])-__FinX
                  __Hits[1]=float(__Hits[1])-__FinY
                  __Hits[2]=float(__Hits[2])-__FinZ

          #Lon Rotate x
          __LongestDistance=0.0
          __Track=__TempTrack[self.LongestTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][0])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Track.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                __Z=float(__hits[2])
                __Pos=float(__hits[0])
                __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                __hits[0]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
          #Lon Rotate y
          __LongestDistance=0.0
          __Track=__TempTrack[self.LongestTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][1])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Track.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
           for __hits in __Tracks:
                __Z=float(__hits[2])
                __Pos=float(__hits[1])
                __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                __hits[1]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
          #Phi rotate print

          __LongestDistance=0.0
          for __Track in __TempTrack:
                __X=float(__Track[len(__Track)-1][0])
                __Y=float(__Track[len(__Track)-1][1])
                __Distance=math.sqrt((__X**2)+(__Y**2))
                if __Distance>=__LongestDistance:
                 __LongestDistance=__Distance
                 __vector_1 = [__Distance, 0]
                 __vector_2 = [__X, __Y]
                 __Angle=-Track.angle_between(__vector_1,__vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                __X=float(__hits[0])
                __Y=float(__hits[1])
                __hits[0]=(__X*math.cos(__Angle)) - (__Y * math.sin(__Angle))
                __hits[1]=(__X*math.sin(__Angle)) + (__Y * math.cos(__Angle))

          #After shift

          __FinZ=666666.0
          for __Track in __TempTrack:
              if __FinZ>float(__Track[0][2]):
                 __FinZ=float(__Track[0][2])-self.Resolution
          __FinX=float(__TempTrack[self.LongestTrackInd][0][0])
          __FinY=float(__TempTrack[self.LongestTrackInd][0][1])
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                 __Hits[0]=float(__Hits[0])-__FinX
                 __Hits[1]=float(__Hits[1])-__FinY
                 __Hits[2]=float(__Hits[2])-__FinZ

          #Rescale
          if Rescale:
              __X=[]
              __Y=[]
              __Z=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __X.append(__hits[0])
                    __Y.append(__hits[1])
                    __Z.append(__hits[2])
              __dUpX=MaxX-max(__X)
              __dDownX=MaxX+min(__X)
              __dX=(__dUpX+__dDownX)/2
              __xshift=__dUpX-__dX
              __X=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[0]=__hits[0]+__xshift
                    __X.append(__hits[0])
            ##########Y
              __dUpY=MaxY-max(__Y)
              __dDownY=MaxY+min(__Y)
              __dY=(__dUpY+__dDownY)/2
              __yshift=__dUpY-__dY
              __Y=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[1]=__hits[1]+__yshift
                    __Y.append(__hits[1])
              __min_scale=max(max(__X)/(MaxX-(2*self.Resolution)),max(__Y)/(MaxY-(2*self.Resolution)), max(__Z)/(MaxZ-(2*self.Resolution)))
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[0]=int(round(__hits[0]/__min_scale,0))
                    __hits[1]=int(round(__hits[1]/__min_scale,0))
                    __hits[2]=int(round(__hits[2]/__min_scale,0))

          #Enchance track
          __TempEnchTrack=[]
          for __Tracks in __TempTrack:
              for h in range(0,len(__Tracks)-1):
                  __deltaX=float(__Tracks[h+1][0])-float(__Tracks[h][0])
                  __deltaZ=float(__Tracks[h+1][2])-float(__Tracks[h][2])
                  __deltaY=float(__Tracks[h+1][1])-float(__Tracks[h][1])
                  try:
                   __vector_1 = [__deltaZ,0]
                   __vector_2 = [__deltaZ, __deltaX]
                   __ThetaAngle=Track.angle_between(__vector_1, __vector_2)
                  except:
                    __ThetaAngle=0.0
                  try:
                    __vector_1 = [__deltaZ,0]
                    __vector_2 = [__deltaZ, __deltaY]
                    __PhiAngle=Track.angle_between(__vector_1, __vector_2)
                  except:
                    __PhiAngle=0.0
                  __TotalDistance=math.sqrt((__deltaX**2)+(__deltaY**2)+(__deltaZ**2))
                  __Distance=(float(self.Resolution)/3)
                  if __Distance>=0 and __Distance<1:
                     __Distance=1.0
                  if __Distance<0 and __Distance>-1:
                     __Distance=-1.0
                  __Iterations=int(round(__TotalDistance/__Distance,0))
                  for i in range(1,__Iterations):
                      __New_Hit=[]
                      if math.isnan(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle)):
                         continue
                      if math.isnan(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle)):
                         continue
                      if math.isnan(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle)):
                         continue
                      __New_Hit.append(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle))
                      __New_Hit.append(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle))
                      __New_Hit.append(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle))
                      __TempEnchTrack.append(__New_Hit)

          #Pixelise print

          self.TrackPrint=[]
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                  __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                  __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                  self.TrackPrint.append(str(__Hits))
          for __Hits in __TempEnchTrack:
                  __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                  __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                  __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                  self.TrackPrint.append(str(__Hits))
          self.TrackPrint=list(set(self.TrackPrint))
          for p in range(len(self.TrackPrint)):
              self.TrackPrint[p]=ast.literal_eval(self.TrackPrint[p])
          self.TrackPrint=[p for p in self.TrackPrint if (abs(p[0])<self.bX and abs(p[1])<self.bY and abs(p[2])<self.bZ)]
          del __TempEnchTrack
          del __TempTrack

      def UnloadTrackPrint(self):
          delattr(self,'TrackPrint')
          delattr(self,'bX')
          delattr(self,'bY')
          delattr(self,'bZ')
          delattr(self,'H')
          delattr(self,'W')
          delattr(self,'L')
          delattr(self,'LongestTrackInd')


      def Plot(self,PlotType):
        if PlotType=='XZ' or PlotType=='ZX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='YZ' or PlotType=='ZY':
          __InitialData=[]
          __Index=-1
          for y in range(-self.bY,self.bY):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.W,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[1])+self.bY][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('Y [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bY,-self.bY])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='XY' or PlotType=='YX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for y in range(-self.bY,self.bY):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.W))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[1]+self.bY)]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Y [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[-self.bY,self.bY,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        else:
          print('Invalid plot type input value! Should be XZ, YZ or XY')


      @staticmethod
      def unit_vector(vector):
          return vector / np.linalg.norm(vector)

      def angle_between(v1, v2):
            v1_u = Track.unit_vector(v1)
            v2_u = Track.unit_vector(v2)
            dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1]      # dot product
            det = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]      # determinant
            return np.arctan2(det, dot)

      def GetEquationOfTrack(Track):
          Xval=[]
          Yval=[]
          Zval=[]
          for Hits in Track:
              Xval.append(Hits[0])
              Yval.append(Hits[1])
              Zval.append(Hits[2])
          XZ=np.polyfit(Zval,Xval,1)
          YZ=np.polyfit(Zval,Yval,1)
          return (XZ,YZ, 'N/A',Xval[0],Yval[0],Zval[0])

      def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
            a0=np.array(a0)
            a1=np.array(a1)
            b0=np.array(b0)
            b1=np.array(b1)
            # If clampAll=True, set all clamps to True
            if clampAll:
                clampA0=True
                clampA1=True
                clampB0=True
                clampB1=True


            # Calculate denomitator
            A = a1 - a0
            B = b1 - b0
            magA = np.linalg.norm(A)
            magB = np.linalg.norm(B)

            _A = A / magA
            _B = B / magB

            cross = np.cross(_A, _B);
            denom = np.linalg.norm(cross)**2


            # If lines are parallel (denom=0) test if lines overlap.
            # If they don't overlap then there is a closest point solution.
            # If they do overlap, there are infinite closest positions, but there is a closest distance
            if not denom:
                d0 = np.dot(_A,(b0-a0))

                # Overlap only possible with clamping
                if clampA0 or clampA1 or clampB0 or clampB1:
                    d1 = np.dot(_A,(b1-a0))

                    # Is segment B before A?
                    if d0 <= 0 >= d1:
                        if clampA0 and clampB1:
                            if np.absolute(d0) < np.absolute(d1):
                                return a0,b0,np.linalg.norm(a0-b0)
                            return a0,b1,np.linalg.norm(a0-b1)


                    # Is segment B after A?
                    elif d0 >= magA <= d1:
                        if clampA1 and clampB0:
                            if np.absolute(d0) < np.absolute(d1):
                                return a1,b0,np.linalg.norm(a1-b0)
                            return a1,b1,np.linalg.norm(a1-b1)


                # Segments overlap, return distance between parallel segments
                return None,None,np.linalg.norm(((d0*_A)+a0)-b0)



            # Lines criss-cross: Calculate the projected closest points
            t = (b0 - a0);
            detA = np.linalg.det([t, _B, cross])
            detB = np.linalg.det([t, _A, cross])

            t0 = detA/denom;
            t1 = detB/denom;

            pA = a0 + (_A * t0) # Projected closest point on segment A
            pB = b0 + (_B * t1) # Projected closest point on segment B


            # Clamp projections
            if clampA0 or clampA1 or clampB0 or clampB1:
                if clampA0 and t0 < 0:
                    pA = a0
                elif clampA1 and t0 > magA:
                    pA = a1

                if clampB0 and t1 < 0:
                    pB = b0
                elif clampB1 and t1 > magB:
                    pB = b1

                # Clamp projection A
                if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                    dot = np.dot(_B,(pA-b0))
                    if clampB0 and dot < 0:
                        dot = 0
                    elif clampB1 and dot > magB:
                        dot = magB
                    pB = b0 + (_B * dot)

                # Clamp projection B
                if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                    dot = np.dot(_A,(pB-a0))
                    if clampA0 and dot < 0:
                        dot = 0
                    elif clampA1 and dot > magA:
                        dot = magA
                    pA = a0 + (_A * dot)


            return pA,pB,np.linalg.norm(pA-pB)

def CleanFolder(folder,key):
    if key=='':
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    else:
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path) and (key in the_file):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
#This function automates csv read/write operations
def LogOperations(flocation,mode, message):
    if mode=='UpdateLog':
        csv_writer_log=open(flocation,"a")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
          log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='StartLog':
        csv_writer_log=open(flocation,"w")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
           log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='ReadLog':
        csv_reader_log=open(flocation,"r")
        log_reader = csv.reader(csv_reader_log)
        return list(log_reader)

def RecCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-TSU'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/REC_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def EvalCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-TSU'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TEST_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def TrainCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-TSU'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TRAIN_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      EOSsubModelDIR=EOSsubDIR+'/'+'Models'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def LoadRenderImages(Tracks,StartTrack,EndTrack):
    import tensorflow as tf
    from tensorflow import keras
    NewTracks=Tracks[StartTrack-1:min(EndTrack,len(Tracks))]
    ImagesY=np.empty([len(NewTracks),1])
    ImagesX=np.empty([len(NewTracks),NewTracks[0].H,NewTracks[0].W,NewTracks[0].L],dtype=np.bool)
    for im in range(len(NewTracks)):
        if hasattr(NewTracks[im],'MC_truth_label'):
           ImagesY[im]=int(float(NewTracks[im].MC_truth_label))
        else:
           ImagesY[im]=0
        BlankRenderedImage=[]
        for x in range(-NewTracks[im].bX,NewTracks[im].bX):
          for y in range(-NewTracks[im].bY,NewTracks[im].bY):
            for z in range(0,NewTracks[im].bZ):
             BlankRenderedImage.append(0)
        RenderedImage = np.array(BlankRenderedImage)
        RenderedImage = np.reshape(RenderedImage,(NewTracks[im].H,NewTracks[im].W,NewTracks[im].L))
        for Hits in NewTracks[im].TrackPrint:
                   RenderedImage[Hits[0]+NewTracks[im].bX][Hits[1]+NewTracks[im].bY][Hits[2]]=1
        ImagesX[im]=RenderedImage
    ImagesX= ImagesX[..., np.newaxis]
    ImagesY=tf.keras.utils.to_categorical(ImagesY,2)
    return (ImagesX,ImagesY)

def SubmitJobs2Condor(job):
    SHName = job[2]
    SUBName = job[3]
    if job[8]:
        MSGName=job[4]
    OptionLine = job[0][0]+str(job[1][0])
    for line in range(1,len(job[0])):
        OptionLine+=job[0][line]
        OptionLine+=str(job[1][line])
    f = open(SUBName, "w")
    f.write("executable = " + SHName)
    f.write("\n")
    if job[8]:
        f.write("output ="+MSGName+".out")
        f.write("\n")
        f.write("error ="+MSGName+".err")
        f.write("\n")
        f.write("log ="+MSGName+".log")
        f.write("\n")
    f.write('requirements = (CERNEnvironment =!= "qa")')
    f.write("\n")
    if job[9]:
        f.write('request_gpus = 1')
        f.write("\n")
    f.write('arguments = $(Process)')
    f.write("\n")
    f.write('+SoftUsed = '+'"'+job[7]+'"')
    f.write("\n")
    f.write('transfer_output_files = ""')
    f.write("\n")
    f.write('+JobFlavour = "workday"')
    f.write("\n")
    f.write('queue ' + str(job[6]))
    f.write("\n")
    f.close()
    TotalLine = 'python3 ' + job[5] + OptionLine
    f = open(SHName, "w")
    f.write("#!/bin/bash")
    f.write("\n")
    f.write("set -ux")
    f.write("\n")
    f.write(TotalLine)
    f.write("\n")
    f.close()
    subprocess.call(['condor_submit', SUBName])
    print(TotalLine, " has been successfully submitted")
