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

But how do they function? Diffusion models utilize 2 main principles the forward diffsuion and the backward diffusion during the forward diffusion we continusoulsy add uniformly distributed noise over T timesteps to our training image x0 till there is only noise left at xT. In the backwards diffusion we try to revert this whole process by substractiong noise from in T uniformly timesteps, here we try to achieve the same noise level as our corresponding point in time xt from the forward diffusion. This is supposed to teach the model how to denoise images or rather noise to create and image.
![illustration forward and backward diffusion process](/images/DiffusionProcessTraining.png)
![inference diffusion](/images/inference%20diffusion.png)



Existing Methods
------
Among existing methods to speed up diffusion model inference there are Solvers and curvature rectification,Reduction of model size and the reduction of sampling steps or 
step distillation.
Solvers and curvature rectification aim to linearize diffusion during inference, naturally when we try to predict the next step in inference or gradient descent having a more linear function will allow us to move further along to the desired value without accuracy loss.
![linearization](/images/linearization.png)
Reducing the model size will reduce the step cost making it cheaper and faster to use, this is a very usefull improvement so long as the performance is similar to a larger model, to ensure this one could use model distillation about which we will learn more later in this blog.

To really scale a diffusion for even faster interference and even real time applications just reducing the model size is not sufficient and we need further improvements such as reducing the sampling step count or step distillation 


Model
------
Imagine flash utilzes 3 main methods for inference speed up there is backwards distillation, shifted reconstruction loss and 
noise correction.


Backward distillation
------
Distillation in machine learning generally refers to the utilization of a larger (teacher) model from which we try to "distill"
knowledge in a smaller (student) model this generally helps to reduce the inference time and reduces memory requirements while still utilizing the knowledge of the larger model.
The paper about Imagine flash introduces a new distillation technique for diffusion models. Which they coined "Backward distillation".

Backward distillation aims to eliminate information leakage from the starting image to the denoising steps during the training phase. The paper suggest this since information leakage reduces
inference performance which becomes especially aparent when only taking a few diffusion steps (small T), which is one of the main ways to decrease inference cost.

To eliminate the information leakage we simulate the inference process during the training phase we achieve this by letting the student model predict the value of xt instead of using the xt that was calculated during the forward diffusion, in doing so we can be sure that none of the original signal x0 is included in our sample xt. This is also the case during the inference process since there is no x0 to source data from. 
[Backward diffusion](/images/backwardDiffusion.png) 
  

b

Noise correction
------
Shifted reconstruction loss
------


Sources
------
https://arxiv.org/abs/2302.02398 https://www.semanticscholar.org/paper/Diffusion-Model-for-Generative-Image-Denoising-Xie-Yuan/6c08f74b8b41cb6f4c36816e81212dc9dbdfadac
https://arxiv.org/pdf/2311.16863 energy consumption
