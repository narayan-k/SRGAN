# SRGAN

### Intro
This is a PyTorch implementation of a super-resolution generative adverserial network according to this [paper](https://arxiv.org/abs/1609.04802). By training this network on the Celeb A HQ [dataset](https://www.kaggle.com/lamsimon/celebahq) the model learns how to generate high quality facial images from lower resolution images.

### General Principles

The SRGAN is essentially made up of three neural networks working together in order to generate fake high resolution images from a low resolution input. The first of the three networks is the untrainable feauture extractor comprising of the VGG-19 network pretrained on imageNet. This extracts features from an input image. The second network is a generator network this takes a low resolution image as input and produces a fake high resolution image as an output. The third network is a discriminator network to distinguish between fake generated images and real high resolution images.

#### Training the generator
A batch of low resolution images is passed to the generator to produce a fake high resolution images. The generator is trained using a combination of two loss functions. First the fake and real high resolution images are passed to the discriminator network to calculate the adverserial loss. Next the pairs of images are passed through the feature extractor network, the extracted features are then used to calculate the content loss.

The total loss is calculated via the equation:
Total Loss = Content Loss + 1e-3 * Adverserial Loss

The generator variables are then updated.

#### Training the discriminator
Training the discriminator is simpler. Labelled real and fake images are passed to the discriminator, the loss is then calculated and the discriminator variables are then updated.


### Results
###### Low Res Image // Generated Image // High Res Image
![batch](images/SRGAN_results.png)

#### References

1. [The paper](https://arxiv.org/abs/1609.04802)
2. [The dataset](https://www.kaggle.com/lamsimon/celebahq) 
3. [A useful implementation](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan)

