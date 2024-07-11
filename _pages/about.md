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
In may researchers at meta published imagine flash a new accleleration method for diffusion models.
Which aims to keep similar quality of the original but significalnty reduces the cost of inference.


Introduction
======


Diffusion models
------
Diffusion models are powerful generative models their impresive performace outshines other model architectures in areas such as image generation.
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
