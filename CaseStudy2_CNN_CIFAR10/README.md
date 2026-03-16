# Case Study 2: CNN Image Classification on CIFAR-10

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

## Dataset

CIFAR-10 contains 60,000 color images of size 32x32 belonging to 10 classes.

Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## CNN Architecture

Input: 32x32x3

Conv2D (32 filters, 3x3)
MaxPooling (2x2)

Conv2D (64 filters, 3x3)
MaxPooling (2x2)

Conv2D (64 filters)

Flatten

Dense (128)

Output Dense (10 Softmax)

## Concepts Used

- Convolutional Neural Networks
- Feature Extraction
- Pooling
- Softmax Classification
- Adam Optimizer

## Run
