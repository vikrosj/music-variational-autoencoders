# music-variational-autoencoders

# MCVAE - Mixture Composer Variational Autoencoder

The MCVAE was created for my thesis "Variational Autoencoders with Mixture Density Networks for Sequence Prediction in Algorithmic Composition". Long title.

At a top level, the thesis asks these two questions:  
*Does music contain a hierarchical component which is relevant when teaching a machine learning model to create music?  
And, can a machine learning model learn long term structure in music, based on its own perception of data?*
  
  
And the short answer two both questions is *yes*.  
  

### The MCVAE is combined by two parts:

1. A variational autoencoder (VAE) for sequence prediction of notes, it is composed by LSTM-layers.
2. A mixture density network (MDN) comprised of a mixture model (MM) and a LSTM-network, to predict *sequences of sequences* of notes.  Sequences of bars, to be clearer.  
  
To test the project, follow all the steps in the notebook *MCVAE full process*.  
To do inference, use the notebook *MCVAE inference*.
