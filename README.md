DeepSource
=======

Installation:

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

Before training, it provides a demaned map from true catalogs. The edge and top view of the demand map is shown below:

<p align="center">
  <img src="./images/ds9flow.jpg" width="800"/>
</p>


<p align="center">
  <img src="./images/Network_1.jpg" width="700"/>
</p>


_Python library for _


**Citing DeepSource:** 
```
@article{Sadr:2018mud,
      author         = "Sadr, A. Vafaei and Vos, Etienne.E. and Bassett, Bruce A.
                        and Hosenie, Zafiirah and Oozeer, N. and Lochner,
                        Michelle",
      title          = "{DeepSource: Point Source Detection using Deep Learning}",
      year           = "2018",
      eprint         = "1807.02701",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.IM",
      SLACcitation   = "%%CITATION = ARXIV:1807.02701;%%"
}
```
