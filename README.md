# ELEC475Lab1

## MLP Autoencoder
A 4 layer MLP Autoencoder designed to be trained on the MNIST dataset. Functions implemented to test trained model include:
- Running through autoencoder
- Denoising
- Linear Interpolation

### Included .py Files
- model.py: Main model code adapted from ELEC475 slides
- train.py: Trains model - designed to be called using terminal (ex.```python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png```)
- lab1_autoencoderTest.py - Uses trained model to output step 4 content
- lab1_addnoise.py - Uses trained model to output step 5 content
- lab1_linearinterpolation.py - Uses trained model to output step 6 content
- lab1.py - Gathers user input for selecting image index/indeces and calls other lab1_xx.py to generate output for all 3. Can be called using terminal (ex. ```python lab1.py –l MLP.8.pth```)

### Also Included 
- various trained models that had a variety of factors changed in order to improve output
- various loss plots
