# deepul-24spring-solutions
Ongoing personal solutions for CS 294-158 Deep Unsupervised Learning Spring 2024.
There are 4 homework assignments in total.

HW1~2 are uploaded.

# Organization of the notebook
Codeblocks in different questions are designed to be executed independently. However, codeblocks in subsequent subquestions may depend on the ones in the previous subquestions.

# HW-1
In this assignment we can get the following exercises:
1. Fit 1D data with softmax/discretized mixture of logistics probability model.
2. Train a simplified version of PixelCNN in the colored Shapes/MNIST dataset.
3. Implement the iGPT on the colored Shapes/MNIST dataset and use K,V caching to accelerate the inference.
4. Use pre-trained VQVAE to quantize the images and train iGPT with them. We can observe quantizing the images significantly decrease the sequence length without compromising the essential information much.
5. Implement causal transformer for text on a small poetry dataset.
6. Take a glimpse of the multimodal model: train causal transformer on a labeled color MNIST dataset.

# HW-2
1. Train and sample from VAE on 2D data.
2. Train and sample VAE on Images on the CIFAR10 dataset. Besides the vanilla VAE, we also need to implement a hierarchical VAE based on the spirit of NVAE.
3. Train a VQ-VAE and train a transformer prior distribution on the latent space on the CIFAR10 dataset. Generate images by sampling from the transformer prior.
