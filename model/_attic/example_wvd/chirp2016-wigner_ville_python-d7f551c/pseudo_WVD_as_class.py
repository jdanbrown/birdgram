# -*- coding: utf-8 -*-
"""
@author: Patrick Balzerowski
"""



from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class PseudoWVD():
    
    def __init__(self,xData,yData):
        self.x=xData
        self.y=yData
        self.already_evaluated=False
    
    def pseudoWVD(self, make_analytic=True):
        '''Calculates the Pseudo-Wigner-Ville Distribution using the analytical signal associated to the real-valued input signal,
        in case the input signal is already complex valued choose 'make_analytic=False'
        '''
        if self.already_evaluated!=True:
            max_t=self.x[-1]
            min_t=self.x[0]
            length_t=len(self.x)
            time_axes=np.linspace(min_t,max_t,length_t)
            time_step=time_axes[1]-time_axes[0]    
            
            
            if make_analytic==True:
                y_analytic=hilbert(self.y)
            else:
                y_analytic=self.y
            
            
            y_analytic_expanded=np.append(np.zeros_like(y_analytic),y_analytic)
            
            
            N=len(y_analytic)
            produkt_matrix=np.ones((len(self.x),2*N),dtype=complex)
            
            for ti in range(0,len(self.x),1):
                produkt_ti=[]
                for k in range(-N+1,N+1,1):
                    produkt_ti.append(y_analytic_expanded[ti+k]*np.conjugate(y_analytic_expanded[ti-k]))  
                produkt_matrix[ti]=np.asarray(2*np.asarray(produkt_ti))
        
            
            FFT_matrix=[]
            freq_matrix=[]
            for zeile in produkt_matrix:
                n=int(len(zeile))        
                if n%2==0:
                    FFT_zeile=1/np.sqrt(n)*np.fft.fft(zeile)[0:int(n/2)]
                    FFT_matrix.append(FFT_zeile)
                    freq_matrix.append(np.fft.fftfreq(n,time_step)[0:int(n/2)])
                    
                elif n%2==1:
                    FFT_zeile=1/np.sqrt(n)*np.fft.fft(zeile)[0:int((n-1)/2)]
                    FFT_matrix.append(FFT_zeile)
                    freq_matrix.append(np.fft.fftfreq(n,time_step)[0:int((n-1)/2)])
        
            
            
            pseudoWVD_matrix=[np.real(zeile) for zeile in FFT_matrix]
            pseudoWVD_freq_axes=np.array(freq_matrix[0])*0.5
            
            self.pseudoWVD_freq_axes=pseudoWVD_freq_axes
            self.time_axes=time_axes
            self.pseudoWVD_matrix=np.transpose(pseudoWVD_matrix)
            self.already_evaluated=True
            
        
    def contour_plot(self, fig_title="default fig_title",axes_titles=["default_x","default_y"],colormap=cm.viridis):
        '''Produces a standard filled contour-plot using matplotlibs contourf
        '''
        plt.figure()
        plt.title(fig_title)
        plt.xlabel(axes_titles[0])
        plt.ylabel(axes_titles[1])
        plt.contourf(self.time_axes,self.pseudoWVD_freq_axes,self.pseudoWVD_matrix,1000,cmap=colormap)
        plt.xlim(self.time_axes[0],self.time_axes[-1])
        plt.ylim(self.pseudoWVD_freq_axes[0],self.pseudoWVD_freq_axes[-1])
        plt.show()
        
        




    