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





Diffusion models
------
Diffusion models are powerful generative models their impressive performace outshines other model architectures in areas such as image generation.
While they are quite great at what they do there are some drawbacks namely in the effieceny department compared to other model-typer such as 
GANs or VAEs. Diffusion models take longer to train often requiering more data, they have generally a higher demand for Hardware resources such as VRAM and 
require more energy to produce results.

But how do they function? Diffusion models utilize 2 main principles the forward diffsuion and the backward diffusion during the forward diffusion we continusoulsy add uniformly distributed noise over T timesteps to our training image x0 till there is only noise left at xT. In the backwards diffusion we try to revert this whole process by substractiong noise from in T uniformly timesteps, here we try to achieve the same noise level as our corresponding point in time xt from the forward diffusion. This is supposed to teach the model how to denoise images or rather noise to create and image.
![illustration forward and backward diffusion process](/images/DiffusionProcessTraining.png)
![inference diffusion](/images/inference%20diffusion.png)



Existing Methods
------
Among existing methods to speed up diffusion model inference there are [Solvers](https://arxiv.org/pdf/2302.04867) and curvature rectification,Reduction of model size and the reduction of sampling steps or 
step distillation.
Solvers and curvature rectification aim to linearize diffusion during inference, naturally when we try to predict the next step in inference or gradient descent having a more linear function will allow us to move further along to the desired value without accuracy loss.
![linearization](/images/linearization.png)
Reducing the model size will reduce the step cost making it cheaper and faster to use, this is a very usefull improvement so long as the performance is similar to a larger model, to ensure this one could use model distillation about which we will learn more later in this blog.

To really scale a diffusion for even faster interference and even real time applications just reducing the model size is not sufficient and we need further improvements such as reducing the sampling step count or step distillation 


Model
------
Imagine flash utilzes 3 main methods for inference speed up there is backwards distillation, shifted reconstruction loss and 
noise correction which are applied to a existing diffusion model in the paper they used [emu](https://ai.meta.com/research/publications/emu-enhancing-image-generation-models-using-photogenic-needles-in-a-haystack/) an image generation diffusion model by meta. Using these methods it can achieve enough of a speed up to allow for real time image inference while the quality is not much below the parent model.

Distillation 
------
When hearing the term distillation most people will probably think of the alcohol industry where a beverage with lower alcohol content gets distilled into a spirit with higher alcohol content but what exaclty does distillation mean and where does it fall in the context of machine learning?
The oxford dictionary defines distillation as "the process or result of getting the essential meaning, ideas or information from something" this definition also applies in the context of machine learning where we try to take the knowledge from a (usually) larger teacher model and distill it into a (usally) smaller student model this is normally done by using the teacher model during the training of the student model. Ideally we are left with a smaller student model that is as accurate as the teacher but signifacntly smaller reducing both memory and performacne costs.

Backward distillation
------

In the paper the researchers introduces a new distillation technique for diffusion models. Which they coined "Backward distillation".

Backward distillation aims to eliminate information leakage from the starting image to the denoising steps during the training phase. The paper suggest this since information leakage reduces
inference performance which becomes especially aparent when only taking a few diffusion steps (small T), which is one of the main ways to decrease inference cost.

To eliminate the information leakage we simulate the inference process during the training phase we achieve this by letting the student model predict the value of x[t](Superscript) instead of using the xt that was calculated during the forward diffusion, in doing so we can be sure that none of the original signal x0 is included in our sample xt. This is also the case during the inference process since there is no x0 to source data from. 
![Backward diffusion](/images/backwardDiffusion.png) 
The new gradients are computed as follows 
![backward distillation](/images/gradientBackwardDiffusion.png)




Shifted reconstruction loss
------
In the iterative process of image generation through diffusion models generally the image composition and overall structure are created first with t closer to T and the details are added later on with t closer to 0. So ideally the student model will learn image composition and overall structure from the teacher when t is closer to T and the details when t is closer to 0.
In order to achieve that the paper introduces Shifted reconstruciont loss (SRL). 

SRL builds up on backward distillation but additionally uses a function that noises the xt -> x0 prediction of the student model to the current t y is designed in a way so that the teacher prioritieses structure and compositon closer to T and details closer to 0, this noised output of the student is then given to the teacher to make a x0 prediction. The new gradients are computed as follows 
![SRLgradient](/images/gradientSRL.png)  

![SRL](/images/SRL.png)

Noise correction
------
To understand noise correction we have to remind ourselves that diffusion models work by predicting noise, so at every timestep xt the diffusion model predicts the noise at xT however there is only noise, therefore predicting noise at xT becomes trivial and we gain nothing of doing so. To remedy this we treat xT as a special case this gives us an additional bias term.
![noise correction](/images/noise%20correction.png)

Comparison to state-of-the-art
------

Quantitative comparison
------
In the paper the researchers first compare Imagine flash to [Step Distillation](https://arxiv.org/pdf/2210.03142), [LCM](https://arxiv.org/pdf/2310.04378)(Latent Consitency Models)  and [ADD](https://arxiv.org/pdf/2311.17042)(Adversarial Diffusion Distillation)​ using [CLIP](https://arxiv.org/pdf/2104.08718), [FID](https://arxiv.org/pdf/1706.08500) and [CompBench](https://arxiv.org/pdf/2307.06350), FID and CLIP measures the image quality and adherence to the prompt, while CompBench is a Benchmark that measures several different image attributes. All the methods were applied to [emu](https://ai.meta.com/research/publications/emu-enhancing-image-generation-models-using-photogenic-needles-in-a-haystack/) an image generation diffusion model.  
![quantitative comparison other methods](/images/comparisonQuantitavieOthers.png)
In the image we can see that Imagine Flash has the best performance out of all the methods listed but it noteably does not beat the baseline which makes sense when we consider that Imagine Flash distills the knowledge of the Teacher model and is therefore very unlikely to outperform it. 

Qualitative comparison
------
![comparison other distillation methods](/images/comparisonOtherMethods.png)
Here we can see imagine flash compared to the other methods in terms of image quality we can observe that imagine flash has sharper images with more detail and vibrant colors.

Comparison other Models
------
![comparison other models](/images/comparisonOtherModels.png)
In this picture we can see Imagine Flash, [Lightning-LDMXL](https://arxiv.org/pdf/2402.13929) and  [LDMXL-Turbo](https://arxiv.org/pdf/2311.17042) compared against their baselines the percentage indicates how good their performance was against the baseline. 

Human evaluation 
------
In the paper they also conducted a human evaluation using 1000 randomly sampled prompts and 5 human annotators.
[human evaluation](/images/humanEvaluation.png)
In the table we can observe the percentages of the annotators on whether they rate the quality of Imagine flash compared to the other listed public models superior, equal or inferior.
We can observe that Imagine Flash seems to win out in most cases in terms of quality.

Ablation
------
In the quantative ablation study we can clearly see that both backwards distillation and SRL have a strong positive impact on the quality of the resulst when regarding the FID amd CLIP score, while noise correction doesnt seem to have an impact at all.
![Ablation](/images/ImagineFlashAblation.png)
In the qualitative ablation we can again observe the impact of backward distillation and SRL on the quality of result, here backward distillation makes the images more crisps with better edges and finer details while SRL adds coherence and structure to the image. This
time we can observe a difference with the usage of Noise correction albeit somewhat minor it becomes apparent that Noise correction leads to the colors becomeing more vibrant and saturated. 
![AlbationQualitative](ImagineFlashAblationQualitative.png)



Conclusion
------
Imagine flash introduces new Methods of applying existing concepts and doing so very succesfully its 3 methods especially SRL and backward diffusion provide a significant quality improvement over comparable methods. They also make significant speedup of diffusion models possible so much so that imagine flash can generate an image while the user is still typing out the prompt.
While imagine flash is great we do have to consider that it will likely not be able too to surpass its teacher model making its performance very dependent on it.

Sources
------
https://arxiv.org/abs/2302.02398 https://www.semanticscholar.org/paper/Diffusion-Model-for-Generative-Image-Denoising-Xie-Yuan/6c08f74b8b41cb6f4c36816e81212dc9dbdfadac
https://arxiv.org/pdf/2311.16863 energy consumption
