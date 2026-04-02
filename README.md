# D2L: Diffuse-to-Latent Distillation for NLOS Wall-Reflected Reconstruction

## Abstract：
Noncontact exfiltration of electronic screen content via optical projection side channels is severely ill-posed: the projection mapping is highly ill-conditioned, making inversion hypersensitive to perturbations, while irreversible light transport removes global semantic cues and amplifies reconstruction ambiguity. We propose a latent-distillation reconstruction framework that avoids explicit inversion by shifting recovery to a stable generative latent space. **Our key innovations are:** **(a)** a **teacher–student latent distillation** scheme where a student regresses latent codes from side-channel observations and a **frozen teacher decoder** acts as a strong generative prior for stable reconstruction; **(b)** a **skip-free teacher autoencoder** (no cross-bottleneck skip connections) that forces reconstructions to be explained by global latent semantics rather than copying corrupted local textures, reducing ambiguity under compression; and **(c)** a lightweight **Global→Local mixing block** that couples **large-kernel depthwise convolution** for long-range aggregation with **small-kernel refinement** for detail recovery. Together, these designs improve stability to projection perturbations and enhance semantic consistency under severe information loss, enabling robust reconstruction of screen content from optical side-channel measurements.
## Overview of the method
![Overview](NLOS_v2.png)
## Performance on datasets
![示例图片](screen.png)
![示例图片](chart.png)
![示例图片](Password.png)
![示例图片](WebSight.png)
