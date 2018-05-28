import numpy as np
import random
import glob
from util import fetch_data,fetch_data_3ch,cat2map

class GeneralDataProvider(object):
    
    """
    CLASS GeneralDataProvider: This class will provide one/multi channel(s) data (image) to feed CCN.
    
    --------
    METHODS:
    
    __init__:
    | arguments:
    |        images_path: list of paths to the images. The string can be produced by glob to make a list of images. 
    |        models_path: list of paths to the images. The string can be produced by glob to make a list of images. 
    |        nx: first dimension size of the provided window (number of pixels).
    |        ny: second dimension size of the provided window (number of pixels).
    |        kernels: List of kernels you want to operate on model image. Model image is a map with same size of image, point sources pixels are filled by one and zero for the rest.). 
    |        margin (default=500): number of pixels you don't want to sample from. Margin will be calculated from borders. For example margin=1000 will not use first and last 1000 pixels of each dimension.)
    |        b_n (default=0.05): background noise. This parameter controls background of demand map.
    |        alpha (default=0.95): this parameter determines contribution of image in demand map.
    |        d_ch: (default=0): The channel you want to get background noise from.
    |        sampling (default='random'): This argument still has one option. leave it as default.
    
    
    __call__:
    This method provides one channel data (image) to feed CCN.
    | Arguments:
    |        n (default=1): number of returned patches.
    |        nx (default=None): first dimension size of the provided window (number of pixels).
    |        ny (default=None): second dimension size of the provided window (number of pixels).
    |        coord (default=False): return coordinate of point sources.
    
    | Returns:
    |        Image, Demand map, coordinates (if coord is true)
    """

    def __init__(self,images_path,models_path,nx,ny,kernels=None,margin=500, b_n=0.05, alpha=0.95, d_ch=0, sampling='random'):

        self.images_path = images_path
        self.models_path = models_path
        self.func = kernels
        self.margin = margin
        self.b_n = b_n
        self.alpha = alpha
        self.d_ch = d_ch
        self.sam_p = sampling
        self.nx,self.ny = nx,ny
        self.n_files = len(images_path)
        assert len(images_path)==len(models_path),'Number of images and model files are different!'
        assert self.n_files>0,'No file found!'
        if type(images_path[0]) is not list :
            self.n_channels = 1
        else:
            self.n_channels = len(images_path[0])
            for i in range(self.n_files):
                assert len(images_path[i])==self.n_channels, 'Path '+str(i)+' has not '+str(self.n_channels)+'channels!'

        if sampling!='random':
            assert len(images_path)==len(sampling),'Number of images and sample files are different!'

        print 'number of files: ',self.n_files,', number of channels: ',self.n_channels

    def __call__(self, n=1, nx=None ,ny=None, coord=False):
    
        margin = self.margin
        i_rand = np.random.randint(self.n_files)

        if nx is None or ny is None:
            nx, ny = self.nx, self.ny

        image_file = self.images_path[i_rand]
        model_file = self.models_path[i_rand]

        if type(image_file) is not list :
            data, x_coords, y_coords = fetch_data(image_file,model_file)
        else:
            data = []
            for i in range(len(image_file)):
                dp, x_coords, y_coords = fetch_data(image_file,model_file)
                data.append(dp)
            data = np.concatenate(data,axis=-1)

        if self.margin!=0:
            for i_ch in range(self.n_channels):
                data[0,:margin,:,i_ch] = 0
                data[0,:,:margin,i_ch] = 0
                data[0,-margin:,:,i_ch] = 0
                data[0,:,-margin:,i_ch] = 0

        lx,ly = data[0,:,:,0].shape

        demand = cat2map(lx,ly,x_coords,y_coords)
        
        if self.func is not None:
            demand = self.func(demand)
        demand = demand-demand.min()
        demand = demand/demand.max()
        demand = (demand+self.b_n)*(data[0,:,:,self.d_ch])**(self.alpha)
        demand = demand-demand.min()
        demand = demand/demand.max()

        X = np.zeros((n, nx, ny, self.n_channels))
        Y = np.zeros((n, nx, ny, self.n_channels))
        if coord:
                crd = [[] for i in range(n)]

        # Random batch
        if self.sam_p=='random':
            for k in range(n):
                i0,j0 = np.random.randint(margin,lx-nx-margin),np.random.randint(margin,ly-ny-margin)
                X[k,:,:,:] = data[0,i0:i0+nx,j0:j0+ny,:]
                Y[k,:,:,0] = demand[i0:i0+nx,j0:j0+ny]

                if coord:
                    for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
                        if i0 <= i < i0+nx and j0 <= j < j0+ny:
                            crd[k].append([i-i0,j-j0])

        else:
            samp_p = np.loadtxt(self.sam_p[i_rand])
            assert samp_p.shape[1]>2,'Sampling pattern have to be either random or a (nx2) numpy array!'
            n_samples = samp_p.shape[0]
            k = 0
            while k<n:
                k_rand = np.random.randint(n_samples)
                i0,j0 = int(samp_p[k_rand,0]),int(samp_p[k_rand,1])
                if i0>margin+nx/2 and lx-i0>margin+nx/2 and j0>margin+ny/2 and ly-j0>margin+ny/2:
                    X[k,:,:,:] = data[0,i0-nx//2:i0+nx//2,j0-ny//2:j0+ny//2,:]
                    Y[k,:,:,0] = demand[i0-nx//2:i0+nx//2,j0-ny//2:j0+ny//2]
                    k += 1

                if coord:
                    for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
                        if i0-nx//2 <= i < i0+nx//2 and j0-ny//2 <= j < j0+ny//2:
                            crd[k].append([i-i0+nx//2,j-j0+ny//2])

        if coord:
                return X, Y, np.array(crd[0])
        else:
                return X, Y

class PreProcessDataProvider(object):
    
    """
    CLASS GeneralDataProvider: This class will provide one channel data (image) to feed CCN.
    
    --------
    METHODS:
    
    __init__:
    | arguments:
    |        files_path: list of paths to the images. The string can be produced by glob to make a list of images. Models have to be in same directory.
    |        nx: first dimension size of the provided window (number of pixels).
    |        ny: second dimension size of the provided window (number of pixels).
    |        kernels: List of kernels you want to operate on model image. Model image is a map with same size of image, point sources pixels are filled by one and zero for the rest.). 
    |        margin (default=1000): number of pixels you don't want to sample from. Margin will be calculated from borders. For example margin=1000 will not use first and last 1000 pixels of each dimension.)
    |        b_n (default=0.05): background noise. This parameter controls background of demand map.
    |        alpha (default=0.95): this parameter determines contribution of image in demand map.
        
    
    __call__:
    This method provides one channel data (image) to feed CCN.
    | Arguments:
    |        n (default=1): number of returned patches.
    |        coord (default=False): return coordinate of point sources.
    
    | Returns:
    |        Image, Demand map, coordinates (if coord is true)
    """

    def __init__(self,files_path,nx,ny,kernels=[],margin=1000, b_n=0.05, alpha=0.95):

          self.files_path = files_path
          self.kernels = kernels
          self.margin = margin
          self.b_n = b_n
          self.alpha = alpha
          self.nx,self.ny = nx,ny
          self.files = glob.glob(files_path)
          self.n_files = len(self.files)
          assert self.n_files>0,'No file found!'
          self.path = "/".join(files_path.split('/')[:-1])+"/"
          print 'number of files: ',self.n_files

    def __call__(self, n=1, coord=False):
          margin = self.margin
          i = np.random.randint(self.n_files)

          image_file = self.files[i]
          model_file = self.path+self.files[i].split('/')[-1].split('_')[1]+'.txt'
          
          data, x_coords, y_coords = fetch_data(image_file,model_file)
          lx,ly = data[0,:,:,0].shape

          cat = cat2map(lx,ly,x_coords,y_coords)
          
          for kernel in self.kernels:
              cat = kernel(cat)

          X = np.zeros((n, self.nx, self.ny, 1))
          Y = np.zeros((n, self.nx, self.ny, 1))
          if coord:
              crd = [[] for i in range(n)]

          # Random batch
          for k in range(n):
              i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
              X[k,:,:,:] = data[:,i0:i0+self.nx,j0:j0+self.ny,:]
              X_min = X[k,:,:,0].min()
              X_max = X[k,:,:,0].max()
              
              y = cat[i0:i0+self.nx,j0:j0+self.ny]
              y = y-y.min()
              y = ((y/y.max())+self.b_n)*(X[k,:,:,0]-X_min)**(self.alpha)
              y = y-y.min()
              y = y/y.max()
              y = y*(X_max-X_min)
              Y[k,:,:,0] = y+X_min
              
              if coord:
                  for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
                      if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
                          crd[k].append([i-i0,j-j0])

          if coord:
              return X, Y, np.array(crd[0])
          else:
              return X, Y
        
class PreProcessDataProvider_3ch(object):
    
    """
    CLASS GeneralDataProvider: This class will provide 3 channels data (image) to feed CCN.
    
    --------
    METHODS:
    
    __init__:
    | arguments:
    |        files_path: list of paths to the images. The string can be produced by glob to make a list of images. Models have to be in same directory.
    |        nx: first dimension size of the provided window (number of pixels).
    |        ny: second dimension size of the provided window (number of pixels).
    |        kernels: List of kernels you want to operate on model image. Model image is a map with same size of image, point sources pixels are filled by one and zero for the rest.). 
    |        margin (default=1000): number of pixels you don't want to sample from. Margin will be calculated from borders. For example margin=1000 will not use first and last 1000 pixels of each dimension.)
    |        b_n (default=0.05): background noise. This parameter controls background of demand map.
    |        alpha (default=0.95): this parameter determines contribution of image in demand map.
        
    
    __call__:
    This method provides one channel data (image) to feed CCN.
    | Arguments:
    |        n (default=1): number of returned patches.
    |        coord (default=False): return coordinate of point sources.
    
    | Returns:
    |        Image, Demand map, coordinates (if coord is true)
    """

    def __init__(self,files_path,nx,ny,kernels,margin=1000, b_n=0.05, alpha=0.95):

          self.files_path = files_path
          self.kernels = kernels
          self.margin = margin
          self.b_n = b_n
          self.alpha = alpha
          self.nx,self.ny = nx,ny
          self.files = glob.glob(files_path)
          self.n_files = len(self.files)
          self.path = "/".join(files_path.split('/')[:-1])+"/"
          print 'number of files: ',self.n_files

    def __call__(self, n=1, coord=False):
          margin = self.margin
          i = np.random.randint(self.n_files)

          image_file = self.files[i]
          model_file = self.path+self.files[i].split('/')[-1].split('_')[1]+'.txt'
                  
          data, x_coords, y_coords = fetch_data_3ch(image_file,model_file)

          lx,ly = data[0,:,:,0].shape

          cat = cat2map(lx,ly,x_coords,y_coords)
          
          for kernel in self.kernels:
              cat = kernel(cat)

          X = np.zeros((n, self.nx, self.ny, 3))
          Y = np.zeros((n, self.nx, self.ny, 1))
          if coord:
              crd = [[] for i in range(n)]

          # Random batch
          for k in range(n):
              i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
              X[k,:,:,0] = data[0,i0:i0+self.nx,j0:j0+self.ny,0]
              X[k,:,:,1] = data[0,i0:i0+self.nx,j0:j0+self.ny,1]
              X[k,:,:,2] = data[0,i0:i0+self.nx,j0:j0+self.ny,2]
              X_min = X[k,:,:,0].min()
              X_max = X[k,:,:,0].max()
              
              y = cat[i0:i0+self.nx,j0:j0+self.ny]
              y = y-y.min()
              y = ((y/y.max())+self.b_n)*(X[k,:,:,0]-X_min)**(self.alpha)
              y = y-y.min()
              y = y/y.max()
    #            y = y*(X_max-X_min)
    #            Y[k,:,:,0] = y+X_min
              Y[k,:,:,0] = y
              
              if coord:
                  for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
                      if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
                          crd[k].append([i-i0,j-j0])

          if coord:
              return X, Y, np.array(crd[0])
          else:
              return X, Y
