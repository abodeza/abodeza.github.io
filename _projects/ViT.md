---
layout: page
title: Image Captioning (ViT) [WIP]
description: Walkthrough
img: assets/img/ViT_image.png
importance: 1
category: computer vision
related_publications: false
---
**Project is being improved**

## Image Captioning with Vision Transformer

This project tackles the problem of image captioning, which involves generating natural language descriptions for input images. Developed initially as the final project for the "Introduction to Deep Learning" course (ECSE 4850), which was then improved to utilize modern cloud-based ML tools. The solution uses a Vision Transformer (ViT) as the image encoder and a Transformer decoder to generate captions. The model is trained and evaluated on the Flickr8K dataset, where each image is associated with five human-written captions.

### Core Components

- **Vision Transformer Encoder**: Processes input images by dividing them into patches, embedding them, and encoding visual features in sequence.
- **Transformer Decoder**: Autoregressively generates captions using encoded visual context and previously generated tokens.
- **Training Pipeline**: Includes batching, token shifting, positional encoding, masking, and evaluation based on BLEU scores.
- **Performance Evaluation**: Uses average BLEU scores across multiple reference captions per image.

The project encourages experimentation with various architectural parameters such as patch size, embedding dimensions, number of attention heads, and encoder depth to improve caption quality. The final model demonstrates the ability to generate coherent and contextually relevant image captions.
