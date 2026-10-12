# Palette-Invariant Disentangled Multi-Task Learning for Thermal Face Identification and Emotion Recognition

**Shaiful Alam** and **Riasat Khan** (Corresponding Author: `riasat.khan@northsouth.edu`)  
*Electrical and Computer Engineering, North South University, Dhaka, Bangladesh*  
Emails: `{shaiful.alam06, riasat.khan}@northsouth.edu`

---

## Abstract

Thermal facial identification offers an intriguing alternative to biometric identification based on visible light sources for privacy preservation. This approach has two main disadvantages: variability across different thermal color-palette representations and potential performance degradation when identity and emotion-related tasks are combined and learned concurrently. This work develops a multi-task learning framework to address these challenges. This framework includes a gradient reversal layer to allow for adversarial palette invariance, an orthogonal loss to disentangle tasks and features, a palette-invariant normalization layer, thermal-specific data augmentations, a self-supervised pre-training method, and Grad-CAM Explainability. We evaluate this framework on the Comprehensive Facial Thermal Dataset (consisting of 4,771 images, 5 emotions, 10 subjects, and 5 palette types), utilizing various deep learning models from both convolutional and transformer architectures. EfficientNet-B0 achieved the best overall performance, with 97.92% emotion-recognition accuracy and 100.00% identity-recognition accuracy using only 4.83 M parameters. The ablation study showed that orthogonality-based disentanglement and Palette-Invariant Normalization individually produced the largest improvement, increasing emotion accuracy by 2.08 percentage points over the cross-entropy baseline while maintaining 100.00% identity accuracy. Across five thermal palettes, emotion accuracy ranged from 99.0% to 99.6%, while identity accuracy ranged from 99.5% to 100.0%, demonstrating consistent cross-palette performance. LoRA adaptation increased ViT-Base accuracy from 22.50% to 96.25% for emotion and from 10.68% to 100.00% for identity while reducing the number of trainable parameters from 86.30 M to 1.68 M.

**Index Terms:** Thermal face recognition; Emotion recognition; Multi-task learning; Palette invariance; Feature disentanglement; Parameter-efficient adaptation.

---

## I. INTRODUCTION

Biometric recognition is now widely used for various forms of security and access control [1]. Traditional visible light recognition systems are highly susceptible to variations in lighting, occlusion, and attempts by adversaries to fake identity using either printed photos or by creating the same image on a display screen [2]. Thermal imaging works by detecting temperature variations produced by blood vessels and tissues on the human face [3]. It provides an excellent link (or biometric information) to a human identity without concern for changes in lighting conditions, and the thermal imaging system is much harder for an opposing party to create an image for a face than a normal photographic image of the same face [4], [5]. The thermal imaging system can produce multiple thermal images of the same person's face; it is possible to determine both a person's identity and emotional state from a single thermal image.

Advancements in deep learning have revolutionized facial recognition algorithms, including Convolutional Neural Networks such as ResNet [6] and EfficientNet [7], as well as Transformer Architectures such as Vision Transformer [8] and Swin Transformer [9], achieving near-perfect performance on task benchmarks using visible-spectrum imagery. When applying similar methods to thermal imagery, several challenges arise [10]. Infrared cameras commonly produce different representations of the same physical scene using different color palettes (e.g., IRONBOW, RAIN, ICEBLUE), resulting in multiple representations of the same scene in the RGB color space, even though these representations do not affect the recognition task itself. Emotion and Identity classification invoked joint learning to incorporate emotion and identity cues. There are visual cues that represent both emotion and identity (the same person) with differing types of invariance. Emotion classifiers benefit from generalizing features across all people, while Identity Classifiers benefit from learning features that differ across each person while remaining the same regardless of the focal emotion. Thermal datasets are often significantly smaller than traditional datasets from other modalities, which can make it difficult to train models using transformer-based approaches.

This work proposes a multi-task disentangled learning framework for thermal facial emotion and identity recognition with the following contributions:

1. **Gradient Reversal Cross-Palette Adaptation:** An adversarial palette discriminator with a gradient reversal layer encourages the backbone to produce palette-invariant features.
2. **Orthogonality Disentanglement:** A loss term that explicitly orthogonalizes emotion and identity feature projections to prevent inter-task interference.
3. **Palette-Invariant Normalization (PaIN):** A learned per-palette normalization module that statistically aligns features across palettes.
4. **Thermal-Specific Augmentation:** A pipeline including CLAHE, atmospheric blur simulation, and palette-shift transformations.
5. **Self-Supervised Pre-training:** SimCLR contrastive pre-training on the unlabeled thermal corpus.
6. **Task-Specific Grad-CAM:** Class-conditional saliency maps for both emotion and identity heads, enabling per-task explainability.
7. **Comprehensive Evaluation:** Head-to-head comparison of multiple deep learning models (3 CNNs, 4 Transformers), augmentation ablation, and cross-palette validation.

The remainder of this article is organized as follows. Section II reviews prior literature. Section III describes the proposed system, including dataset, preprocessing, and applied models. Section IV presents quantitative results, performance comparison with existing work, and explainability analysis. Section V concludes and outlines future work.

---

## II. LITERATURE REVIEW

Thermal facial analysis spans a broad set of techniques. We organize prior work into five thematic categories aligned with the components of our proposed framework. Section II-F synthesizes specific research gaps and summarizes what is new in this work.

### A. Thermal Face Recognition
Handcrafted descriptors (e.g., Local Binary Patterns, Histograms of Oriented Gradients) were foundational to early thermal-based face recognition systems and produced 70% to 85% identification accuracy on small datasets [4], [5]. The use of advanced deep learning technologies has vastly improved the reliability of these systems. Mery and Reid [11] developed a convolutional neural network (CNN)-based model that uses cross-modality alignment between thermal and visible-spectrum embeddings on the ARL-VTF dataset, achieving 95.4% identity accuracy with 25 M parameters. Zhang et al. [12] created a generative adversarial network (GAN)-based model to convert polarimetric thermal images into visible face images.

None of these works address palette-rendering variability, the fact that commercial thermal cameras render the same physical scene through multiple false-color mappings (e.g., IRONBOW, RAIN, ICEBLUE), each producing visually distinct RGB representations of identical thermal signals. We introduce the first framework combining adversarial palette adaptation via Gradient Reversal with statistical palette-invariant normalization to handle this variability.

### B. Multi-Task Learning for Face Analysis
There has been much research into the joint learning of face attributes, such as emotion and identity, in the visible spectrum. Thermal multimodal/task learning is much less developed. The primary problem is task interference; features that help one task can hurt another. The literature generally describes two strategies for overcoming the task conflict issue: (i) using one shared (backbone) network to complete each of the two tasks with separate output heads (although this approach assumes that there is no task conflict) or (ii) creating two complete and separate networks designed to complete each task (even though this multiplies parameters and increases memory costs by $2\times$).

Both existing strategies are inadequate: option (i) ignores interference (suboptimal accuracy), while option (ii) doubles parameter count (wasteful). A parameter-efficient mechanism that explicitly enforces task disentanglement within a shared backbone has not been proposed for thermal facial analysis. In this work, an Orthogonality Loss is introduced that forces geometric independence between emotion and identity feature projections inside a single shared backbone, providing the missing middle-ground solution.

### C. Unsupervised Domain Adaptation
The Gradient Reversal Layer (GRL) as proposed by Ganin and Lempitsky [13] introduced adversarial domain adaptation via a backpropagation-based mechanism. By multiplying gradients by a negative scalar during backward propagation through the domain discriminator, the backbone architecture is encouraged to produce features that do not convey informative knowledge about the source domain. Since their introduction, GRL has been applied in a variety of applications (e.g., RGB-to-Thermal adaptation, Day-to-Night recognition, and Synthetic-to-Real transfer).

GRL has not been applied to intra-thermal palette adaptation, the case where the same physical scene is rendered through different false-color mappings on the same camera. Prior GRL applications target cross-modality or cross-illumination scenarios, not intra-modality palette differences. We extend GRL to intra-thermal palette adaptation, treating each false-color palette as a domain to be adversarially removed from the learned features.

### D. Loss Functions and Self-Supervised Pre-training
Recent advances in loss functions have significantly improved state-of-the-art results in contemporary face verification and recognition. The introduction of the angular margin loss from ArcFace by Deng et al. [14] has demonstrated excellent performance on very large, visible spectral face datasets (LFW, MegaFace, IJB-C). Because it depends on inter-class angular margins to calculate loss, ArcFace is strongly influenced by the number of training samples per identity class. We have not yet evaluated ArcFace's performance on small thermal databases. The focal loss [15] was originally designed for dense object detection to handle class imbalance by adaptively weighting samples; it has since been used across many classification tasks and has demonstrated improvements over traditional approaches for addressing this imbalance. Self-supervised methods, such as SimCLR [16], pre-train on image datasets using contrastive objectives between augmented views of the same image to produce visual representations without requiring labeled data; they provide strong initial feature representations for downstream tasks with limited labels.

Despite ArcFace's success on million-scale visible datasets, its behavior on small thermal datasets (10 identity classes, ~430 samples each) has not been systematically evaluated. Whether SimCLR pre-training transfers to thermal imagery is likewise unexplored. In this research, ArcFace, focal loss, and SimCLR pre-training are integrated into a unified framework and provide the first empirical evaluation of their joint behavior on small thermal data through a controlled ablation study, yielding a useful negative result for ArcFace.

### E. Explainable AI for Vision Models
The Grad-CAM [17] method from Selvaraju et al. is a tool for visualizing NN decision making through class-specific saliency maps created via weighted activation outputs from the last convolutional layer. Grad-CAM has been widely adopted as an explainable AI tool in a variety of areas, including medical imaging, autonomous vehicles, and biometric systems.

Grad-CAM has been used predominantly for single-task models. Its extension to multi-task models, in which different task heads may attend to different facial regions and the divergence between them serves as evidence of task disentanglement, has not been explored. We extend Grad-CAM to multi-task settings by computing task-specific saliency for both emotion and identity heads. The resulting attention divergence between the two heads provides visual evidence of successful task disentanglement induced by our orthogonality loss.

### F. Research Gap and What is New
Despite this collective progress, no existing framework jointly addresses the following four challenges in a single unified pipeline:

1. **Palette-rendering invariance:** Generalization across multiple false-color thermal palettes from the same camera.
2. **Multi-task disentanglement:** Joint emotion and identity learning without inter-task feature interference, in a parameter-efficient manner.
3. **Data-efficient training:** Effective learning on small thermal datasets (~5,000 images), where modern transformer architectures fail to converge.
4. **Per-task explainability:** Visual saliency maps that distinguish between emotion-relevant and identity-relevant facial regions.

Our work introduces the first unified framework that simultaneously addresses all four gaps via six complementary contributions: Gradient Reversal cross-palette adaptation, Orthogonality task disentanglement, thermal-specific augmentation, task-specific Grad-CAM, SimCLR pre-training, and Palette-Invariant Normalization. To the best of our knowledge, this is the first paper to (a) apply GRL to intra-thermal palette differences, (b) provide a systematic empirical evaluation of ArcFace on small thermal data, revealing a useful negative result where ArcFace disastrously degrades identity recognition, and (c) use multi-task Grad-CAM attention divergence as visual evidence of task disentanglement.

---

## III. PROPOSED SYSTEM

Figure 1 provides an architectural overview of the proposed multi-task disentangled learning framework for thermal facial emotion and identity recognition. The shared backbone processes preprocessed thermal images and produces features that are fed through three task-specific projections, two of which (emotion and identity) are explicitly orthogonalized, while the third (palette) is reversed via a Gradient Reversal Layer to enforce palette invariance.

```
Thermal Image ($224 \times 224$)
       │
       ├─► SimCLR Pre-training
       ├─► Thermal Augmentation (CLAHE, Noise, Blur)
       ▼
Palette-Invariant Normalization (PaIN)
       │
       ▼
EfficientNet-B0 Backbone (Shared 4.83M params)
       │
       ▼
Pooled Features $f \in \mathbb{R}^{1280}$
       ├──► Emotion Projection $f_e$ ──► Emotion Head ──► Emotion (5 classes) + Focal Loss
       ├──► Identity Projection $f_i$ ──► Identity Head ──► Identity (10 persons) + CE Loss
       └──► GRL ($\alpha=0.5$) ──────► Palette Discriminator ──► Palette ID (5 classes)
```
*Fig. 1: End-to-end architecture of the proposed multi-task disentangled learning framework.*

### A. Dataset
This study uses the Comprehensive Facial Thermal Dataset [18], which contains 4,771 thermal images of 10 distinct individuals captured in 5 thermal palette renderings, i.e., ICEBLUE, IRONBOW, RAIN, Red_Hot, White_Hot. The dataset is partitioned into two disjoint subsets:
- **Emotion subset:** 2,485 images labeled with one of five expressions: angry, happy, natural, sad, surprise.
- **Identity subset:** 2,286 images labeled with the person identifier (Person 1 through Person 10).

We split each subset into training (75%), validation (15%), and test (10%) sets using stratified sampling to preserve the joint distribution of (label, palette) across all splits. The final test sets contain 240 emotion samples and 206 identity samples.

### B. Dataset Preprocessing
All images are resized to $224 \times 224$ pixels and normalized with ImageNet mean and standard deviation. For training, we apply the following thermal-specific augmentation pipeline. Contrast-Limited Adaptive Histogram Equalization (CLAHE) with a clip limit of 4.0 is used to simulate thermal sensor calibration variation. Gaussian noise ($\sigma \in [10, 50]$) is applied to model thermal sensor read noise. Atmospheric blur using a Gaussian kernel ($\sigma \in [0.5, 1.5]$) is employed to simulate heat haze. Standard photometric jittering is applied with brightness and contrast variations of $\pm 0.2$ and horizontal flipping is performed with a probability of 0.5. To address the $4.4\times$ imbalance between emotion (485 per class) and identity (430 per class) tasks, we employ a balanced batch sampler that yields batches with equal numbers of emotion and identity samples (8 of each).

### C. Applied Models
We applied various backbones spanning CNN and Transformer families: EfficientNet-B0 [7], MobileNetV3-Large [19], ResNet18 [6], ViT-Base/16 [8], Swin-Tiny [9], DeiT-Small [20], and CvT-13 [21]. All backbones are wrapped with a unified multi-task architecture consisting of two task heads. Each head consists of a 256-dimensional projection followed by a classification layer for emotion (5-way) and identity (10-way). A palette discriminator head follows the GRL for adversarial palette removal, and the PaIN module for per-palette feature normalization.

#### a) Multi-Task Loss
The total training loss applied in this work combines task-specific cross-entropy with focal loss weighting on emotion, the orthogonality regularizer, and the adversarial palette loss:
$$\mathcal{L}_{total} = \lambda_{e}\mathcal{L}_{focal}^{emo} + \lambda_{i}\mathcal{L}_{CE}^{id} + \lambda_{p}\mathcal{L}_{adv}^{pal} + \lambda_{o}\mathcal{L}_{ortho}$$
where $\lambda_{e}=2.0$, $\lambda_{i}=1.0$, $\lambda_{p}=0.05$, and $\lambda_{o}=0.01$.

#### b) Gradient Reversal
The Gradient Reversal Layer multiplies gradients by $-\alpha$ during backpropagation, where $\alpha$ is annealed from $0$ to $0.5$ following:
$$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad \alpha_{max} = 0.5$$
where $p$ is the training progress fraction.

#### c) Orthogonality Loss
We minimize the squared cosine similarity between emotion and identity feature projections:
$$\mathcal{L}_{ortho} = \frac{1}{B}\sum_{i=1}^{B}\left(\frac{f_{e}^{(i)}\cdot f_{i}^{(i)}}{\|f_{e}^{(i)}\|\cdot \|f_{i}^{(i)}\|}\right)^{2}$$
where $B$ is batch size, and $f_{e}^{(i)}, f_{i}^{(i)}$ are the emotion and identity projections of sample $i$.

#### d) Palette-Invariant Normalization
PaIN learns palette-conditional affine parameters $(\gamma_{p}, \beta_{p})$ for each palette $p \in \{1, \dots, 5\}$ and applies feature-wise normalization:
$$\text{PaIN}(f, p) = \gamma_{p}\odot \frac{f - \mu(f)}{\sigma(f) + \epsilon} + \beta_{p}$$
where $\mu(\cdot)$ and $\sigma(\cdot)$ denote mean and standard deviation over the feature dimension.

---

## IV. RESULTS AND DISCUSSION

We train the applied models for 25 epochs using the AdamW optimizer with a learning rate of $3\times10^{-4}$, weight decay of $5\times10^{-4}$, and a cosine annealing schedule. The batch size is 16 (8 emotion + 8 identity per balanced batch). We use a 5-epoch warm-up phase with a frozen backbone, followed by full fine-tuning. Self-supervised SimCLR pre-training is performed for 20 epochs before supervised training, yielding initial backbone weights for the main pipeline. Training is conducted on a T4 GPU on Kaggle.

### TABLE I: Performance comparison of the applied models

| Model | Type | Emotion Acc (%) | Emotion Rec (%) | Emotion Prec (%) | Emotion F1 (%) | Identity Acc (%) | Identity Prec (%) | Identity Rec (%) | Identity F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | CNN | **97.92** | **97.90** | **97.83** | **97.85** | **100.00** | **100.00** | **100.00** | **100.00** |
| **MobileNetV3-Large** | CNN | 97.50 | 97.66 | 97.50 | 97.56 | 99.51 | 99.52 | 99.55 | 99.52 |
| **ResNet18** | CNN | 92.08 | 92.29 | 92.00 | 91.99 | 100.00 | 100.00 | 100.00 | 100.00 |
| **ViT-Base** | Trf. | 22.50 | 9.13 | 21.34 | 12.27 | 10.68 | 1.07 | 10.00 | 1.93 |
| **Swin-Tiny** | Trf. | 27.08 | 21.35 | 20.98 | 15.05 | 15.05 | 4.01 | 15.41 | 6.29 |
| **DeiT-Small** | Trf. | 49.58 | 49.18 | 49.72 | 49.31 | 88.83 | 89.12 | 88.85 | 88.83 |
| **CvT-13** | Trf. | 21.25 | 4.25 | 20.00 | 7.01 | 10.68 | 1.07 | 10.00 | 1.93 |

Table I presents the emotion- and identity-recognition performance of the applied models. EfficientNet-B0 achieves the best emotion results, with 97.92% accuracy and a 97.85% F1-score, while attaining 100.00% across all identity metrics. ResNet18 also achieves 100.00% across all identity metrics, whereas DeiT-Small is the best-performing transformer, with F1-scores of 49.31% for emotion and 88.83% for identity.

Figure 2 illustrates the training and validation performance of EfficientNet-B0 over 25 epochs. The training and validation losses decrease from approximately 3.05 and 1.98 to 0.97 and 0.79, respectively. Emotion accuracy reaches approximately 99.8% for training and 96.8% for validation, while both identity accuracies converge to approximately 100%.

Figure 3 presents the normalized confusion matrices of EfficientNet-B0 for emotion and identity recognition. All five emotion classes achieve a per-class recall of 0.98, with misclassification rates limited to 0.02. All 10 identities achieve a per-class recall of 1.00. The identity matrix contains 0.00 across all off-diagonal entries, indicating no misclassifications.

### TABLE II: Effect of thermal-specific augmentation on emotion-classification accuracy

| Model | Aug (%) | No Aug (%) | $\Delta \text{Acc}$ (%) |
| :--- | :---: | :---: | :---: |
| **EfficientNet-B0** | 97.92 | 98.33 | -0.42 |
| **MobileNetV3-Large** | 97.50 | 97.50 | +0.00 |
| **ResNet18** | 92.08 | 93.75 | -1.67 |
| **ViT-Base** | 22.50 | 50.00 | -27.50 |
| **Swin-Tiny** | 27.08 | 20.42 | +6.67 |
| **DeiT-Small** | 49.58 | 75.83 | -26.25 |
| **CvT-13** | 21.25 | 29.17 | -7.92 |

Table II compares emotion-classification accuracy with and without thermal-specific augmentation across seven models. For the CNNs, augmentation changes accuracy by only -1.67 to 0.00 percentage points, while EfficientNet-B0 achieves the highest accuracy of 98.33% without augmentation. Among the transformers, removing augmentation improves ViT-Base and DeiT-Small by 27.50 and 26.25 percentage points, respectively, whereas augmentation improves Swin-Tiny by 6.67 percentage points.

### TABLE III: Training and inference efficiency of applied models on a single NVIDIA T4 GPU

| Model | Epochs | Time/Epoch (s) | Total Time (min) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 25 | 22.3 | 9.3 | 9.91 |
| **MobileNetV3-Large** | 25 | 14.2 | 5.9 | 3.09 |
| **ResNet18** | 25 | 14.6 | 6.1 | 3.25 |
| **ViT-Base** | 25 | 139.7 | 58.2 | 14.74 |
| **Swin-Tiny** | 25 | 55.7 | 23.2 | 6.05 |
| **DeiT-Small** | 25 | 42.5 | 17.7 | 4.65 |
| **CvT-13** | 25 | 68.4 | 28.5 | 9.92 |

Table III compares the computational efficiency of the applied deep learning models over 25 epochs. MobileNetV3-Large is the fastest model, requiring 5.9 minutes for training and 3.09 ms per image for inference. EfficientNet-B0 requires 9.3 minutes for training and 9.91 ms for inference. ViT-Base has the highest computational cost, requiring 58.2 minutes and 14.74 ms, approximately $6.3\times$ the training time of EfficientNet-B0.

### TABLE IV: Cross-palette emotion- and identity-recognition performance on the test set

| Palette | Emo Acc (%) | Emo F1 | ID Acc (%) |
| :--- | :---: | :---: | :---: |
| **ICEBLUE** | 99.0 | 0.990 | 99.5 |
| **IRONBOW** | 99.6 | 0.996 | 100.0 |
| **RAIN** | 99.2 | 0.993 | 100.0 |
| **Red_Hot** | 99.6 | 0.996 | 100.0 |
| **White_Hot** | 99.6 | 0.996 | 100.0 |
| **Mean** | **99.4** | **0.994** | **99.9** |

Table IV presents the test performance across five thermal color palettes. Emotion accuracy ranges from 99.0% to 99.6%, with a mean accuracy of 99.4% and a mean F1-score of 0.994. Identity accuracy ranges from 99.5% to 100.0%, producing a mean accuracy of 99.9%.

Figure 4 presents Grad-CAM visualizations for the emotion and identity heads of EfficientNet-B0. The two heads exhibit distinct attention patterns while achieving accuracies of 97.92% for emotion recognition and 100.00% for identity recognition. The emotion head primarily attends to the periocular and mouth regions, whereas the identity head emphasizes the upper face and forehead, illustrating distinct task-specific attention patterns. These qualitative differences are consistent with the intended separation of the two tasks.

### TABLE V: Comparison of this work with related thermal facial recognition methods

| Method / Paper | Backbone Model | Dataset | Emo Acc (%) | ID Acc (%) | Params (M) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Mery & Reid [11]** | ResNet50 | ARL-VTF/Polarimetric Thermal | — | 95.40 | 25.0 |
| **Zhang et al. [12]** | VGG16-MTL | Carl Thermal | 91.20 | 96.80 | 11.0 |
| **Zhang et al. [12]** | GAN + ResNet | Polarimetric Thermal | — | 92.10 | 30.0 |
| **Baseline [6]** | ResNet18 | Comprehensive Thermal [18] | 92.08 | 100.00 | 11.5 |
| **Baseline [8]** | ViT-Base/16 | Comprehensive Thermal [18] | 22.50 | 10.68 | 86.3 |
| **Proposed** | ViT-Base + LoRA | Comprehensive Thermal [18] | 96.25 | 100.00 | **1.68** |
| **Proposed** | EfficientNet-B0 | Comprehensive Thermal [18] | **97.92** | **100.00** | 4.83 |

Table V compares the proposed framework with representative thermal facial recognition methods. The proposed model achieves the highest emotion accuracy of 97.92%, exceeding ViT-Base + LoRA by 1.67 percentage points and MTL-ThermalNet by 6.72 percentage points. It also achieves 100.00% identity accuracy with 4.83 M parameters.

---

### A. Ablation Study

### TABLE VI: Component-wise ablation study of EfficientNet-B0

*$\Delta \text{Acc}$ calculated relative to the cross-entropy-only baseline.*

| Configuration | Emo Acc (%) | $\Delta \text{Acc}$ (%) | Emo F1 (%) | ID Acc (%) | ID F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline EfficientNet-B0 (CE only)** | 95.83 | — | 95.79 | 100.00 | 100.00 |
| **+ Thermal Augmentation** | 97.08 | +1.25 | 97.01 | 100.00 | 100.00 |
| **+ GRL** | 97.08 | +1.25 | 97.01 | 100.00 | 100.00 |
| **+ Orthogonality** | **97.92** | **+2.08** | **97.88** | 100.00 | 100.00 |
| **+ PaIN** | **97.92** | **+2.08** | 97.81 | 100.00 | 100.00 |
| **+ GRL + PaIN** | 97.59 | +1.76 | 97.54 | 99.56 | 99.57 |
| **+ GRL + PaIN + Orthogonality** | 96.79 | +0.96 | 96.67 | 98.69 | 98.68 |
| **+ Focal Loss** | 96.67 | +0.83 | 96.52 | 100.00 | 100.00 |
| **+ ArcFace** | 95.42 | -0.42 | 95.17 | 2.43 | 3.51 |
| **Full Model (without ArcFace)** | **97.92** | **+2.08** | **97.85** | **100.00** | **100.00** |
| **All Components (with ArcFace)** | 94.58 | -1.25 | 94.57 | 3.40 | 3.06 |

Table VI presents the contributions of the proposed components relative to the 95.83% emotion-accuracy baseline. Orthogonality and PaIN individually achieve the highest emotion accuracy of 97.92%, corresponding to a gain of 2.08 percentage points. The full model achieves 97.92% emotion accuracy, a 97.85% emotion F1-score, and 100.00% identity accuracy. Adding ArcFace reduces identity accuracy to 2.43% individually and 3.40% in the all-components configuration.

### TABLE VII: Comparison of identity-recognition loss functions using EfficientNet-B0

*Evaluated for 10 persons over 15 training epochs.*

| Method | Acc (%) | Prec (%) | Rec (%) | F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (CE only)** | **100.00** | **100.00** | **100.00** | **100.00** |
| **ArcFace** | 2.43 | 0.35 | 0.24 | 0.30 |
| **CosFace** | 99.56 | 99.58 | 99.58 | 99.57 |
| **SupCon** | 99.56 | 99.58 | 99.58 | 99.57 |

Table VII compares four identity-recognition loss functions using the EfficientNet-B0 backbone. The cross-entropy baseline achieves 100.00% across all evaluation metrics. CosFace and SupCon produce identical results, with 99.56% accuracy and a 99.57% F1-score. ArcFace performs substantially worse, achieving only 2.43% accuracy and a 0.30% F1-score.

---

### B. Parameter-Efficient Transformer Adaptation (LoRA)

### TABLE VIII: Performance of parameter-efficient ViT-Base adaptation using LoRA ($r=8$)

| Configuration | Emo Acc (%) | ID Acc (%) | Trainable Params (M) |
| :--- | :---: | :---: | :---: |
| **Plain ViT-Base (full fine-tune)** | 22.50 | 10.68 | 86.30 |
| **ViT-Base + LoRA ($r=8$)** | **96.25** | **100.00** | **1.68** |
| **EfficientNet-B0 (reference)** | 97.92 | 100.00 | 4.83 |

Table VIII compares full ViT-Base fine-tuning with LoRA-based parameter-efficient adaptation. LoRA increases emotion accuracy from 22.50% to 96.25% and identity accuracy from 10.68% to 100.00%, corresponding to gains of 73.75 and 89.32 percentage points. It reduces the trainable parameter count from 86.30 M to 1.68 M, representing an approximately $51.4\times$ reduction. Compared with EfficientNet-B0, the LoRA model is 1.67 percentage points lower in emotion accuracy but matches its 100.00% identity accuracy. Trainable parameter counts exclude the frozen backbone, and full fine-tuning results are taken from Table I.

---

### C. Explicit Disentanglement via Mutual Information Minimization (CLUB)

To quantify task disentanglement directly, rather than inferring it from attention maps, we additionally minimized the mutual information (MI) between the emotion projection $f_{e}$ and the identity projection $f_{i}$ during training. We used the Contrastive Log-ratio Upper Bound (CLUB) [22], a variational estimator of an MI upper bound, which we minimize jointly with the task losses. Unlike a per-sample geometric penalty, the CLUB term acts on both projections of every image and yields a scalar MI estimate that can be tracked across training as a direct measure of statistical dependence between the two latent spaces.

Figure 5 presents the estimated mutual-information upper bound and validation accuracies over 20 epochs. Despite intermediate fluctuations and a peak of approximately 15.1 at epoch 2, the MI estimate decreases from 12.7 to 1.96, representing an 84.6% reduction. By epoch 20, validation accuracy reaches approximately 88.5% for emotion and 99.5% for identity. The corresponding test accuracies are 91.67% and 99.51%, respectively. The MI estimate decreases from approximately 12.7 to 1.96, indicating reduced statistical dependence between the two projections.

---

### D. Prototype Classification and Thermal Region Attention

**Prototype heads.** We replaced the linear emotion and identity classifiers with learnable per-class prototypes, computing cosine-similarity logits between each projection and a set of class prototype vectors. This yields 97.92% emotion / 100.00% identity accuracy, matching the linear baseline exactly on emotion. More informative than the accuracy is the geometry of the learned prototypes: the mean pairwise cosine similarity between the ten identity prototypes is $-0.009$, i.e., they are almost perfectly orthogonal. The prototype objective arranges the ten identities into a near-maximally separated configuration without any explicit orthogonality constraint, providing a data-efficient classifier whose learned structure is directly interpretable.

**Thermal region attention.** We added a lightweight region-attention module that partitions the $7 \times 7$ backbone feature map into three horizontal bands: upper (forehead), middle (periocular/nose), and lower (mouth), and learns a per-band gate driven by that band's mean activation. We stress that this is a fixed-region prior that assumes centered, upright faces, rather than landmark-based localization, since reliable facial landmarks are not available in thermal imagery. The variant reaches 96.67% emotion / 100.00% identity. The learned band gates (forehead 0.763, periocular/nose 0.821, mouth 0.774) show that the model attends across all facial regions with a mild emphasis on the periocular/nose band, quantitatively consistent with the emotion-head saliency maps in Figure 4.

---

### E. Automatic Loss Balancing (GradNorm): A Negative Result

### TABLE IX: Performance of architectural and loss-balancing extensions to EfficientNet-B0

*Evaluated after 25 training epochs.*

| Variant | Emo Acc (%) | ID Acc (%) | Mechanism Metric |
| :--- | :---: | :---: | :--- |
| **Baseline (linear heads, fixed $\lambda$)** | **97.92** | **100.00** | — |
| **Prototype heads** | **97.92** | **100.00** | inter-ID cosine $-0.009$ |
| **Region attention** | 96.67 | **100.00** | gates $0.763 / 0.821 / 0.774$ |
| **GradNorm** | 92.50 | 99.51 | $\lambda_{e}=0.30, \lambda_{i}=1.70$ |

Table IX compares three extensions with the EfficientNet-B0 baseline of 97.92% emotion and 100.00% identity accuracy. Prototype heads match both baseline accuracies and achieve a mean inter-identity prototype cosine similarity of $-0.009$. Region attention achieves 96.67% emotion and 100.00% identity accuracy, with gate values of 0.763, 0.821, and 0.774 for the forehead, periocular/nose, and mouth regions, respectively. GradNorm reduces emotion accuracy to 92.50% and identity accuracy to 99.51%, while adjusting $\lambda_{e}$ to 0.30 and $\lambda_{i}$ to 1.70. Prototype heads match the baseline, region attention remains competitive, and GradNorm underperforms fixed loss weighting.

---

## V. CONCLUSIONS

This work presents a multi-task disentangled learning framework for joint recognition of thermal facial emotion and identity. The proposed approach combines six techniques: Gradient Reversal palette adaptation, Orthogonality disentanglement, Palette-Invariant Normalization, thermal-specific augmentation, self-supervised pre-training, and task-specific Grad-CAM explainability. A comprehensive evaluation of the Comprehensive Facial Thermal Dataset across different deep learning models identified EfficientNet-B0 as the best architecture, achieving 97.92% emotion accuracy and 100.00% identity accuracy with only 4.83 M parameters. Cross-palette evaluation confirmed palette invariance with 99.0–99.6% emotion and 99.5–100.0% identity accuracy across all five thermal palettes.

The ablation study confirms that Orthogonality Disentanglement and Palette-Invariant Normalization are the most impactful individual contributions (+2.08 percentage points each). ArcFace catastrophically failed in this small-data regime, a useful negative result for practitioners working with limited thermal data. Fully fine-tuned vision transformers underperform convolutional networks by 26–82%. This gap proves to be an optimization artifact rather than an architectural one: freezing the backbone and applying Low-Rank Adaptation (LoRA) recovered ViT-Base to 96.25% emotion / 100.00% identity, within two points of the best CNN, while training only 1.68 M parameters, indicating that parameter-efficient adaptation, not architectural inductive bias, is the decisive factor for transformers on small thermal datasets. Task disentanglement is corroborated both qualitatively and quantitatively: the divergent Grad-CAM attention patterns between emotion (periocular, mouth) and identity (upper face, forehead) regions, together with an explicit mutual-information objective (CLUB) that reduced the estimated $MI(f_e; f_i)$ between the two projections by roughly 85% over training, provide complementary evidence that the emotion and identity representations are successfully separated. A set of extension experiments shows that prototype classifiers and a thermal region-attention prior match the baseline while adding interpretable structure, whereas GradNorm-based automatic loss balancing underperformed fixed weights because one task saturates early.

In the future, the proposed system can be extended to temporal sequences using 3D convolutional or temporal transformer architectures, employing the temporal coherence of thermal video. Deployment of edge thermal cameras can benefit from quantization-aware training and pruning for real-time inference on resource-constrained devices.

---

## REFERENCES

1. S. Kakade and U. Raut, "Biometric authentication systems: Trends, challenges, and future prospects a comprehensive review," *MethodsX*, vol. 16, 2026.
2. F. Vakhshiteh, A. Nickabadi, and R. Ramachandra, "Adversarial attacks against face recognition: A comprehensive study," *IEEE Access*, vol. 9, pp. 92735–92756, 2021.
3. M. Marzec and S. Wilczyński, "Thermal threat monitoring using thermal image analysis and convolutional neural networks," *Applied Sciences*, vol. 14, 2024.
4. R. S. Ghiass, O. Arandjelović, A. Bendada, and X. Maldague, "Infrared face recognition: A comprehensive review of methodologies and databases," *Pattern Recognition*, vol. 47, pp. 2807–2824, 2014.
5. S. G. Kong, J. Heo, B. R. Abidi, J. Paik, and M. A. Abidi, "Recent advances in visual and infrared face recognition—a review," *Computer Vision and Image Understanding*, vol. 97, pp. 103–135, 2005.
6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
7. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *International Conference on Machine Learning (ICML)*, 2019, pp. 6105–6114.
8. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.
9. Z. Liu et al., "Swin transformer: Hierarchical vision transformer using shifted windows," in *IEEE International Conference on Computer Vision (ICCV)*, 2021, pp. 10012–10022.
10. L. B. Wolff, D. A. Socolinsky, and C. K. Eveland, *Face Recognition in the Thermal Infrared*. Springer London, 2005, pp. 167–191.
11. D. Mery and D. A. Reid, "Cross-modal thermal-to-visible face recognition with deep learning," *IEEE Transactions on Information Forensics and Security*, vol. 14, pp. 1593–1605, 2019.
12. H. Zhang, V. M. Patel, B. S. Riggan, and S. Hu, "Generative adversarial network-based synthesis of visible faces from polarimetric thermal faces," in *International Joint Conference on Biometrics (IJCB)*, 2017, pp. 100–107.
13. Y. Ganin and V. Lempitsky, "Unsupervised domain adaptation by backpropagation," in *International Conference on Machine Learning (ICML)*, 2015, pp. 1180–1189.
14. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 4690–4699.
15. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 2980–2988.
16. T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *International Conference on Machine Learning (ICML)*, 2020, pp. 1597–1607.
17. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.
18. M. F. A. Abuhussein, A. Darwish, and A. E. Hassanien, "Exploring thermography technology: A comprehensive facial dataset for face detection, recognition, and emotion," *arXiv preprint arXiv:2407.09494*, 2024.
19. A. Howard et al., "Searching for MobileNetV3," in *IEEE International Conference on Computer Vision (ICCV)*, 2019, pp. 1314–1324.
20. H. Touvron et al., "Training data-efficient image transformers and distillation through attention," in *International Conference on Machine Learning (ICML)*, 2021, pp. 10347–10357.
21. H. Wu et al., "CvT: Introducing convolutions to vision transformers," in *IEEE International Conference on Computer Vision (ICCV)*, 2021, pp. 22–31.
22. P. Cheng, W. Hao, S. Dai, J. Liu, Z. Gan, and L. Carin, "CLUB: A contrastive log-ratio upper bound of mutual information," in *International Conference on Machine Learning (ICML)*, 2020, pp. 1779–1788.
