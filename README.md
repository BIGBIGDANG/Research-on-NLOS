# D2L: Diffuse-to-Latent Distillation for NLOS Wall-Reflected Reconstruction

## Abstract

Recovering hidden electronic screen content from wall-mediated diffuse reflections in non-line-of-sight (NLOS) scenes is a highly challenging inverse problem. Unlike conventional image restoration, the observed wall patterns are formed through indirect light transport, where multi-bounce propagation, diffuse scattering, and severe energy attenuation jointly destroy the spatial correspondence between the hidden screen and the captured measurement. This leads to two fundamental challenges: **(i)** the mapping from screen content to wall observation is highly ill-conditioned, making direct inversion extremely sensitive to perturbations and noise; and **(ii)** diffuse light transport irreversibly removes global semantic structure and high-frequency details, resulting in severe reconstruction ambiguity. To address these issues, we propose a latent-distilled generative reconstruction framework for passive NLOS screen recovery, with three key innovations: **(a)** **a bottleneck-constrained teacher autoencoder**, whose decoder reconstructs screen content exclusively from compact latent codes without cross-scale feature bypass and is frozen as a strong generative prior; **(b)** **a teacher-student latent regression scheme**, in which a student encoder maps wall-reflection measurements into the latent space and is optimized with joint latent-alignment and image-reconstruction objectives to encourage semantically faithful recovery; and **(c)** **a lightweight Global-to-Local feature mixing block**, which combines large-kernel depthwise aggregation for long-range semantic modeling with local refinement for structural detail recovery. By replacing unstable image-space inversion with latent-space regression constrained by a frozen decoder prior, our framework simultaneously improves reconstruction stability and semantic consistency under severe information loss. Extensive experiments demonstrate that the proposed method enables robust recovery of hidden screen images from passive wall-reflected light in challenging NLOS settings.
## Overview of the method
![Overview](NLOS_v2.png)
## Performance on datasets
![示例图片](screen.png)
![示例图片](chart.png)
![示例图片](Password.png)
![示例图片](WebSight.png)
