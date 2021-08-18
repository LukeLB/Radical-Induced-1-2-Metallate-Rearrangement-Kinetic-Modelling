# BCB boronate model #8

A scipt used to implement a kinetic model for a radical-induced 1,2-metallate rearrangement. 
It works by defining a system of differential equations which can be solved by numerical integration. The model can
then be fit onto the observed kinetic data through least squares. This script was implemented because of the difficulty 
of using standard kinetic modelling software packages with spectrally derived data. Hence, the least squares fit also 
includes parameters to fit spectral feature signal strength in the data which are unknown.

## Usage

Simply run the script to create and fit the model. This will print out visual fitted plots of the model which are used in 
Figure 5 of the main manuscript. Make sure you have also downloaded the observed kinetic data csv files.

## Contributing
You are welcome to use the code here to implement your own models.

## License
[MIT]
