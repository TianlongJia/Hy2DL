# Hy<sup>2</sup>DL: Hybrid Hydrological modeling using Deep Learning methods
![#](docs/source/_static/Hy2DL.jpg)

<p align="justify">
Hy<sup>2</sup>DL is a python library to create hydrological models for rainfall-runoff prediction using deep learning methods. Besides the main code, the repository provides implementation examples using the main Large-Sample-Hydrology datasets (e.g. CAMELS_GB, CAMELS_US, CAMELS_DE, ...). Besides data-driven architectures, the repository also features hybrid hydrological models.

The logic of the codes presented here is heavily based on 'NeuralHydrology --- A Python library for Deep Learning research in hydrology' (https://github.com/neuralhydrology/neuralhydrology.git). For a more flexible, robust and modular implementation of deep learning method in hydrological modeling we advice the use of Neural Hydrology.

## Structure of the repository:
The codes presented in the repository are in the form of python scripts. Additionally several experiments are in the form of JupyterNotebooks for easy reproduction and execution. Following is a quick overview of the repository structure:
- **benchmarks**: Comparison of our library against other studies from scientific literature.
- **data**: Folder where the different datasets (e.g CAMELS-GB, CAMELS-US...) should be added. This information should be independently downloaded by the user.
- **docs**: Library documentation
- **hy2dl**: Code of the library.
- **notebooks**: Jupyter notebooks showing implementation examples, for different cases.
- **results**: Folder where the results generated by the codes will be stored.

## Documentation:
Detailed documentation for the repository can be found at [Hy2DL.readthedocs.io](https://hy2dl.readthedocs.io/en/latest/index.html). 

## Citation:
This code started as part of our study:

```
@Article{hess-28-2705-2024,
AUTHOR = {Acu\~na Espinoza, E. and Loritz, R. and \'Alvarez Chaves, M. and B\"auerle, N. and Ehret, U.},
TITLE = {To bucket or not to bucket? Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization},
JOURNAL = {Hydrology and Earth System Sciences},
VOLUME = {28},
YEAR = {2024},
NUMBER = {12},
PAGES = {2705--2719},
URL = {https://hess.copernicus.org/articles/28/2705/2024/},
DOI = {10.5194/hess-28-2705-2024}
}
```

## Original authors:
 - Eduardo Acuña Espinoza (eduardo.espinoza@kit.edu)
 - Ralf Loritz (ralf.loritz@kit.edu)
 - Manuel Álvarez Cháves (manuel.alvarez-chaves@simtech.uni-stuttgart.de)


 ## Disclaimer:
 No warranty is expressed or implied regarding the usefulness or completeness of the information and documentation provided. References to commercial products do not imply endorsement by the Authors. The concepts, materials, and methods used in the algorithms and described in the documentation are for informational purposes only. The Authors has made substantial effort to ensure the accuracy of the algorithms and the documentation, but the Authors shall not be held liable, nor his employer or funding sponsors, for calculations and/or decisions made on the basis of application of the scripts and documentation. The information is provided "as is" and anyone who chooses to use the information is responsible for her or his own choices as to what to do with the data. The individual is responsible for the results that follow from their decisions.

This web site contains external links to other, external web sites and information provided by third parties. There may be technical inaccuracies, typographical or other errors, programming bugs or computer viruses contained within the web site or its contents. Users may use the information and links at their own risk. The Authors of this web site excludes all warranties whether express, implied, statutory or otherwise, relating in any way to this web site or use of this web site; and liability (including for negligence) to users in respect of any loss or damage (including special, indirect or consequential loss or damage such as loss of revenue, unavailability of systems or loss of data) arising from or in connection with any use of the information on or access through this web site for any reason whatsoever (including negligence).