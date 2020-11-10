DeepSource
=======

**Installation:**

The project is hosted on GitHub. Get a copy by running:
```
$ git clone https://github.com/vafaei-ar/deepsource.git
```
The package is tested on [**Anaconda**](https://www.anaconda.com/download/#linux). You need to have the packages listed in requirements.txt or you can install them using:
```
$ pip install -r requirements.txt 
```
Install the package using:
```
$ cd deepsource

$ python setup.py install
```

**DeepSource** is a flexible and expendable point source detection package for radio telescope images. It takes simulated images and catalogs to train a neural network as signal to noise magnifier. Then it can provide a catalog of predicted point sources by thresholding bob detection (TBD).

<p align="center">
  <img src="./images/ds9flow.jpg" width="800"/>
</p>


<p align="center">
  <img src="./images/Network_1.jpg" width="700"/>
</p>


_Python library for _


**Citing DeepSource:** 
```  
@article{vafaei2019deepsource,
      title={DEEPSOURCE: point source detection using deep learning},
      author={Vafaei Sadr, A and Vos, Etienne E and Bassett, Bruce A and Hosenie, Zafiirah and Oozeer, N and Lochner, Michelle},
      journal={Monthly Notices of the Royal Astronomical Society},
      volume={484},
      number={2},
      pages={2793--2806},
      year={2019},
      publisher={Oxford University Press}
}
```
