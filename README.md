# Awesome_Saliency_prediction

## 全景图像显著性预测

## 2024:

### 1. 

**Authors**: 

<details span>
<summary><b>Abstract</b></summary>
</details>


  [📄 Paper](https://arxiv.org/pdf/2401.02436.pdf) 

## 2023:

### 1. CASP-Net: Rethinking Video Saliency Prediction from an Audio-Visual Consistency Perceptual Perspective

**Authors**: Junwen Xiong, Ganglai Wang , Peng Zhang, Wei Huang, Yufei Zha, Guangtao Zhai

<details span>
<summary><b>Abstract</b></summary>
Incorporating the audio stream enables Video Saliency Prediction (VSP) to imitate the selective attention mechanism of human brain. By focusing on the benefits of joint auditory and visual information, most VSP methods are capable of exploiting semantic correlation between vision and audio modalities but ignoring the negative effects due to the temporal inconsistency of audio-visual intrinsics. Inspired by the biological inconsistency-correction within multi-sensory information, in this study, a consistencyaware audio-visual saliency prediction network (CASPNet) is proposed, which takes a comprehensive consideration of the audio-visual semantic interaction and consistent perception. In addition a two-stream encoder for elegant association between video frames and corresponding sound source, a novel consistency-aware predictive coding is also designed to improve the consistency within audio and visual representations iteratively. To further aggregate the multi-scale audio-visual information, a saliency decoder is introduced for the final saliency map generation. Substantial experiments demonstrate that the proposed CASP-Net outperforms the other state-of-the-art methods on six challenging audio-visual eye-tracking datasets. For a demo of our system please see our project webpage.
</details>



  [📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xiong_CASP-Net_Rethinking_Video_Saliency_Prediction_From_an_Audio-Visual_Consistency_Perceptual_CVPR_2023_paper.html) | [🌐 Project Page]() | [💻 Code](https://woshihaozhu.github.io/CASP-Net/) | [🎥 Short Presentation]()



## 图像视频显著性预测

## 2024:

## 2023:

## 2022 and pre:

### 1. TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection

<details span>
<summary><b>Abstract</b></summary>
TASED-Net is a 3D fully-convolutional network architecture for video saliency detection. It consists of two building blocks: first, the encoder network extracts low-resolution spatiotemporal features from an input clip of several consecutive frames, and then the following prediction network decodes the encoded features spatially while aggregating all the temporal information. As a result, a single prediction map is produced from an input clip of multiple frames. Frame-wise saliency maps can be predicted by applying TASED-Net in a sliding-window fashion to a video. The proposed approach assumes that the saliency map of any frame can be predicted by considering a limited number of past frames. The results of our extensive experiments on video saliency detection validate this assumption and demonstrate that our fully-convolutional model with temporal aggregation method is effective. TASED-Net significantly outperforms previous state-of-the-art approaches on all three major large-scale datasets of video saliency detection: DHF1K, Hollywood2, and UCFSports. After analyzing the results qualitatively, we observe that our model is especially better at attending to salient moving objects.   
</details>

  [📄 Paper]([[1908.05786\] TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection (arxiv.org)](https://arxiv.org/abs/1908.05786))| [🌐 Project Page]() | [💻 Code]([MichiganCOG/TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection (ICCV 2019) (github.com)](https://github.com/MichiganCOG/TASED-Net))

### 2. GASP: Gated Attention for Saliency Prediction

<details span>
<summary><b>Abstract</b></summary>
Saliency prediction refers to the computational task of modeling overt attention. Social cues greatly influence our attention, consequently altering our eye movements and behavior. To emphasize the efficacy of such features, we present a neural model for integrating social cues and weighting their influences. Our model consists of two stages. During the first stage, we detect two social cues by following gaze, estimating gaze direction, and recognizing affect. These features are then transformed into spatiotemporal maps through image processing operations. The transformed representations are propagated to the second stage (GASP) where we explore various techniques of late fusion for integrating social cues and introduce two subnetworks for directing attention to relevant stimuli. Our experiments indicate that fusion approaches achieve better results for static integration methods, whereas non-fusion approaches for which the influence of each modality is unknown, result in better outcomes when coupled with recurrent models for dynamic saliency prediction. We show that gaze direction and affective representations contribute a prediction to ground-truth correspondence improvement of at least 5% compared to dynamic saliency models without social cues. Furthermore, affective representations improve GASP, supporting the necessity of considering affect-biased attention in predicting saliency.    
</details>

  [📄 Paper]([[2206.04590\] GASP: Gated Attention For Saliency Prediction (arxiv.org)](https://arxiv.org/abs/2206.04590)) | [🌐 Project Page]() | [💻 Code]([fabawi/gasp-gated-attention-for-saliency-prediction: GASP: Gated Attention for Saliency Prediction (IJCAI-21) (github.com)](https://github.com/fabawi/gasp-gated-attention-for-saliency-prediction))

