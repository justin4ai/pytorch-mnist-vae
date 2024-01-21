## pytorch-mnist-vae

This is a repository for well-explained VAE template code for MNIST generation in Pytorch.

Corresponding Velog post has not been published yet, so please be waiting for it!

## Environment

With the given ```Dockerfile```, please use configuration settings of ```.devcontainer/devcontainer.json```.

## Train

```bash
python3 train.py
```

Note : You have to change some of code lines if you want to make different directory dependencies.

## Inference

```bash
python3 inference.py
```

Note : You have to change some of code lines if you want to make different directory dependencies.

## Output

<p align = "center"><img src="./ground_truth/mnist_bce2.png"> <br> Ground-truth images</p>

<p align = "center"><img src="./recon/mnist_bce2.png"> <br> Reconstructed images</p>

<p align = "center"><img src="./generated/mnist_bce2.png"> <br> Generated images from noises</p>