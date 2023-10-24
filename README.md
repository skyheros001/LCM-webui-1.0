![info](https://github.com/skyheros001/LCM-webui-1.0/assets/32533832/9bb6221a-8055-4950-a3eb-d2b89efebba9)

### 部署：
1. pip install -r requirements.txt
2. python app.py


相关文章：
https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
https://latent-consistency-models.github.io/

"潜在一致性模型"（Latent Consistency Models，LCMs）的生成模型，这些模型被视为是"潜在扩散模型"（Latent Diffusion Models，LDMs）之后的下一代生成模型。LDMs已经在合成高分辨率图像方面取得了显著的成果，但由于其需要进行迭代采样，因此计算开销大，生成速度较慢。

受"一致性模型"的启发，作者提出了LCMs，这些模型可以在任何预训练的LDM上进行快速的生成，而且只需要很少的迭代步骤，包括Stable Diffusion。LCMs将引导反扩散过程视为解决潜在空间中的增强概率流ODE（PF-ODE），因此可以直接预测这种ODE的解，从而减少了多次迭代的需求，实现了快速、高保真度的采样。

另外类似于lora，相应也的有"潜在一致性微调"（Latent Consistency Fine-tuning，LCF）的新方法，用于在自定义图像数据集上微调LCMs。对LAION-5B-Aesthetics数据集的评估结果表明，LCMs在进行少步骤推断时实现了文本到图像生成性能的最新水平。
