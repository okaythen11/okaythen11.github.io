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
In may researchers at meta published imagine flash a new accleleration [method](https://arxiv.org/pdf/2405.05224) for diffusion models called Imagine Flash [1]. Diffusion models have
seen a surge in popularity lately which is due to their performance in generative tasks such as image generation or resolution upscaling.

The method(s) aim to keep similar quality to the original model but with significantly lower inference cost.
Why is lowering the (energy) cost of model inference important you may ask? Well a study[2] conducted by Hugging Face and Carnegie Mellon University found that generating
1 image can consume as much electricity as it takes to charge a phone this is unsutainable for large scale use especially when considering the iterative process (prompt adjustment) 
people often employ when working with models several outputs are generated till the user is satisfied with the result.





Diffusion models
------
Diffusion models[3] are powerful generative models their impressive performace outshines other model architectures in areas such as image generation.
While they are quite great at what they do there are some drawbacks namely in the effieceny department compared to other model-types such as 
GANs[4] or VAEs[5]. Diffusion models take longer to train often requiering more data, they have generally a higher demand for Hardware resources such as VRAM and 
require more energy to produce results.

But how do they function? Diffusion models utilize 2 main principles the forward diffsuion and the backward diffusion during the forward diffusion we continusoulsy add uniformly distributed noise over T timesteps to our training image x0 till there is only noise left at xT. In the backwards diffusion we try to revert this whole process by substractiong noise from in T uniformly timesteps, here we try to achieve the same noise level as our corresponding point in time xt from the forward diffusion. This is supposed to teach the model how to denoise images or rather noise to create and image.
![illustration forward and backward diffusion process](/images/DiffusionProcessTraining.png)
Forward and backward diffusion
![inference diffusion](/images/inference%20diffusion.png)
Inference (only forward diffusion)



Existing Methods
------
Among existing methods to speed up diffusion model inference there are [Solvers](https://arxiv.org/pdf/2302.04867)[6] and curvature rectification[7],Reduction of model size[8] and the reduction of sampling steps or 
step distillation[9].
Solvers and curvature rectification aim to linearize diffusion during inference, naturally when we try to predict the next step in inference or gradient descent having a more linear function will allow us to move further along to the desired value without accuracy loss.

<img src="/images/linearization.png" alt="Description of Image 1" width="600" height="300">

Reducing the model size will reduce the step cost making it cheaper and faster to use, this is a very usefull improvement so long as the performance is similar to a larger model, to ensure this one could use model distillation about which we will learn more later in this blog.

To really scale a diffusion for even faster interference and even real time applications just reducing the model size is not sufficient and we need further improvements such as step distillation. 

Step distillation however shows serious quality degredation when reducing the sampling steps too much which warrants additional training enhancements such as ADD[10], UFO-GEN[11] or Lightning[12].  


Model
------
Imagine flash utilzes 3 main methods for inference speed up there is backwards distillation, shifted reconstruction loss and 
noise correction which are applied to a existing diffusion model in the paper they used emu[13] an image generation diffusion model by meta. Using these methods it can achieve enough of a speed up to allow for real time image inference while the quality is not much below the parent model.

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
![backward distillation](/images/gradientBackwardDiffusionModified.png)
![backward distillation](/images/gradientBackwardDiffusionModified2.png)




Shifted reconstruction loss
------
In the iterative process of image generation through diffusion models generally the image composition and overall structure are created first with t closer to T and the details are added later on with t closer to 0. So ideally the student model will learn image composition and overall structure from the teacher when t is closer to T and the details when t is closer to 0.
In order to achieve that the paper introduces Shifted reconstruciont loss (SRL). 

SRL builds up on backward distillation but additionally uses a function that noises the xt -> x0 prediction of the student model to the current t y is designed in a way so that the teacher prioritieses structure and compositon closer to T and details closer to 0, this noised output of the student is then given to the teacher to make a x0 prediction.
<img src="/images/SRL.png" alt="Description of Image 1" width="600" height="300">
 The new gradients are computed as follows 
![SRLgradient](/images/gradientSRL.png)  



Noise correction
------
To understand noise correction we have to remind ourselves that diffusion models work by predicting noise, so at every timestep xt the diffusion model predicts the noise at xT however there is only noise, therefore predicting noise at xT becomes trivial and we gain nothing of doing so. To remedy this we treat xT as a special case this gives us an additional bias term.
![noise correction](/images/noise%20correction.png)

Comparison to state-of-the-art
------

Quantitative comparison
------
In the paper the researchers first compare Imagine flash to Step Distillation[9], [LCM](https://arxiv.org/pdf/2310.04378)(Latent Consitency Models)  and [ADD](https://arxiv.org/pdf/2311.17042)[10](Adversarial Diffusion Distillation)​ using [CLIP](https://arxiv.org/pdf/2104.08718)[14], [FID](https://arxiv.org/pdf/1706.08500)[15] and [CompBench](https://arxiv.org/pdf/2307.06350)[16], FID and CLIP measures the image quality and adherence to the prompt, while CompBench is a Benchmark that measures several different image attributes. All the methods were applied to [emu](https://ai.meta.com/research/publications/emu-enhancing-image-generation-models-using-photogenic-needles-in-a-haystack/) an image generation diffusion model.  
![quantitative comparison other methods](/images/comparisonQuantitavieOthers.png)
In the image we can see that Imagine Flash has the best performance out of all the methods listed but it noteably does not beat the baseline which makes sense when we consider that Imagine Flash distills the knowledge of the Teacher model and is therefore very unlikely to outperform it. 

Qualitative comparison
------
![comparison other distillation methods](/images/comparisonOtherMethods.png)
Here we can see imagine flash compared to the other methods in terms of image quality we can observe that imagine flash has sharper images with more detail and vibrant colors.

Comparison other Models
------
<img src="/images/comparisonOtherModels.png" alt="Description of Image 1" width="600" height="300">

In this picture we can see Imagine Flash, [Lightning-LDMXL](https://arxiv.org/pdf/2402.13929)[17] and  [LDMXL-Turbo](https://arxiv.org/pdf/2311.17042)[18] compared against their baselines the percentage indicates how good their performance was against the baseline. 

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
![AlbationQualitative](/images/ImagineFlashAblationQualitative.png)



Conclusion
------
Imagine flash introduces new Methods of applying existing concepts and doing so very succesfully its 3 methods especially SRL and backward diffusion provide a significant quality improvement over comparable methods. They also make significant speedup of diffusion models possible so much so that imagine flash can generate an image while the user is still typing out the prompt.
While imagine flash is great we do have to consider that it will likely not be able too to surpass its teacher model making its performance very dependent on it.

Future work should focus on finding ways to further enhance the baseline diffsuion model thus uplifting the quality of the resulsts, further reduction of step cost and model size are also an option to make inference even faster and cheaper. 

References
------
1. [Imagine Flash: Accelerating Emu Diffusion Models with Backward Distillation, Jonas Kohler, Albert Pumarola, Edgar Schönfeld, Artsiom Sanakoyeu, Roshan Sumbaly, Peter Vajda and Ali Thabet](https://arxiv.org/abs/2405.05224) (2024)

2. [Power Hungry Processing: Watts Driving the Cost of AI Deployment?, Alexandra Sasha Luccioni, Yacine Jernite, Emma Strubell](https://arxiv.org/abs/2311.16863) (2024)
3. [Diffusion Models: A Comprehensive Survey of Methods and Applications, Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, Ming-Hsuan Yang](https://arxiv.org/pdf/2209.00796)
4. [Generative Adversarial Networks: An Overview Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, Anil A Bharath](https://arxiv.org/abs/1710.07035)
5. [Tutorial on Variational Autoencoders Carl Doersch](https://arxiv.org/abs/1606.05908)
6. [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu](https://arxiv.org/abs/2302.04867) (2023)
7. [Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and diffusions] (https://arxiv.org/abs/2303.08797) arXiv preprint (2023)
8. [Li, Y., Wang, H., Jin, Q., Hu, J., Chemerys, P., Fu, Y., Wang, Y., Tulyakov, S.,
Ren, J.: Snapfusion: Text-to-image diffusion model on mobile devices within two
seconds](https://arxiv.org/abs/2306.00980) arXiv preprint (2023)
9. [Meng, C., Rombach, R., Gao, R., Kingma, D., Ermon, S., Ho, J., Salimans, T.: On
distillation of guided diffusion models](https://arxiv.org/abs/2210.03142) In: Proceedings of the IEEE/CVF Confeence on Computer Vision and Pattern Recognition. pp. 14297–14306 (2023)
10. [Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion 
distillation] (https://arxiv.org/abs/2311.17042) arXiv preprint (2023)
11. [Xu, Y., Zhao, Y., Xiao, Z., Hou, T.: Ufogen: You forward once large scale text-toimage generation via diffusion gans] arXiv preprint arXiv:2311.09257 (2023)
12. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929 (2024)
13. https://ai.meta.com/research/publications/emu-enhancing-image-generation-models-using-photogenic-needles-in-a-haystack/
14. [Hessel, J., Holtzman, A., Forbes, M., Bras, R.L., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning] arXiv preprint arXiv:2104.08718 (2021)
15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained
by a two time-scale update rule converge to a local nash equilibrium. Advances in
neural information processing systems  (2017)
16. Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive
benchmark for open-world compositional text-to-image generation. Advances in
Neural Information Processing Systems (2024)
17. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929 (2024)

18. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042 (2023)





