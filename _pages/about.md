---
permalink: /
title: "Imagine Flash"
author_profile: false
redirect_from: 
  - /about/
  - /about.html
---
This is a blog about imagine flash created for a university seminar.



Imagine Flash
======
In may researchers at meta published imagine flash a new accleleration method for diffusion models, which have 
seen a surge in popularity lately which is due to their performance in generative tasks such as image generation or resolution upscaling.
Which aims to keep similar quality of the original but significalnty reduces the cost of inference.
Why is it important to lower the (energy) cost of model inference you may ask a study conducted by Hugging Face and Carnegie Mellon University found that generating
1 image can consume as much electricity as it takes to charge a phone.

Why is reducing costs 
======


Diffusion models
------
Diffusion models are powerful generative models their impressive performace outshines other model architectures in areas such as image generation.
While they are quite great at what they do there are some drawbacks namely in the effieceny department compared to other model-typer such as 
GANs or VAEs. Diffusion models take longer to train often requiering more data, they have generally a higher demand for Hardware resources such as VRAM and 
require more energy to produce resulsts.
But how do they function? Diffusion models utilize 2 main principles the forward diffsuion and the backward diffusion

Existing Methods
------
Among existing methods to speed up diffusion model inference there are Solvers and curvature rectification,Reduction of model size and the reduction of sampling steps then there is also 
step distillation


Model
------
Imagine flash utilzes 3 main methods for inference speed up there is backwards distillation, noise correction 
and shifted reconstruction loss​.

Backwards distillation
------
Distillation in machine learning generally refers to the utilization of a larger (teacher) model from which we try to "distill"
knowledge in a smaller (student) model this generally helps to reduce the inference time and reduces memory requirements
b

Noise correction
------
Shifted reconstruction loss
------


Sources
------
https://arxiv.org/abs/2302.02398 https://www.semanticscholar.org/paper/Diffusion-Model-for-Generative-Image-Denoising-Xie-Yuan/6c08f74b8b41cb6f4c36816e81212dc9dbdfadac
https://arxiv.org/pdf/2311.16863 energy consumption
