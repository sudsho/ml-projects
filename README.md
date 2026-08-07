# ML & Data Science Projects

Collection of machine learning and data science projects exploring different techniques, datasets, and real-world problems.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [Conformal Prediction for Distribution-Free Uncertainty](./conformal-prediction-intervals/) | Split and full conformal from scratch in NumPy - nonconformity scores on a held-out calibration set, the (n+1)-corrected quantile giving finite-sample marginal coverage for any base model, adaptive intervals via conformalized quantile regression | 🔄 In Progress |
| [Hidden Markov Model with Forward-Backward and Viterbi](./hmm-forward-backward/) | Discrete HMM in NumPy - forward/backward recursions in log space, smoothed posteriors, Viterbi decoding with backpointers, Baum-Welch EM with a monotonicity check, label-permutation alignment and a restart study | ✅ Complete |
| [Gaussian Process Regression from Scratch](./gaussian-process-regression/) | Exact GP regression in NumPy - RBF / Matern 3/2 / periodic kernels, posterior via Cholesky solve, hyperparameters by log marginal likelihood, predictive uncertainty bands vs a ridge baseline | ✅ Complete |
| [RealNVP Normalizing Flow on MNIST](./normalizing-flow-mnist/) | Normalizing flow from scratch in PyTorch - dequantization + logit preprocessing, affine coupling layers with alternating masks, exact change-of-variables log-likelihood, temperature sampling and slerp latent interpolation | ✅ Complete |
| [Gradient Boosting Regressor from Scratch](./gradient-boosting-from-scratch/) | Gradient boosting machine in NumPy - CART regression-tree weak learner, pseudo-residual boosting loop with shrinkage, robust losses, stochastic subsampling and early stopping, gain-weighted feature importance, benchmarked against scikit-learn | ✅ Complete |
| [Byte-Pair Encoding Tokenizer from Scratch](./bpe-tokenizer/) | Subword BPE tokenizer in pure Python - corpus prep, greedy most-frequent-pair merges, encoder/decoder with round-trip tests | ✅ Complete |
| [Word2Vec Skip-Gram with Negative Sampling](./word2vec-skipgram/) | Skip-gram word embeddings from scratch - subsampling, dynamic-window pairs, negative sampling, analogy + t-SNE evaluation | ✅ Complete |
| [Titanic Survival Prediction](./titanic-survival-prediction/) | Classic ML classification - EDA, feature engineering, model comparison | ✅ Complete |
| [Sentiment Analysis on Product Reviews](./sentiment-analysis-reviews/) | NLP project using TF-IDF and deep learning for sentiment classification | ✅ Complete |
| [House Price Prediction](./house-price-prediction/) | Regression with advanced feature engineering on California Housing data | ✅ Complete |
| [Image Classification with CNNs](./image-classification-cifar/) | Deep learning - CNN from scratch and transfer learning on CIFAR-10 | ✅ Complete |
| [Customer Segmentation with Clustering](./customer-segmentation/) | Unsupervised learning - K-Means, DBSCAN, hierarchical clustering on customer data | ✅ Complete |
| [Stock Price Forecasting](./time-series-stock-prediction/) | Time series analysis with ARIMA, Prophet, and LSTM for stock price prediction | ✅ Complete |
| [Movie Recommendation System](./recommendation-system/) | Collaborative filtering and content-based recommendations on MovieLens | ✅ Complete |
| [Credit Card Fraud Detection](./fraud-detection/) | Imbalanced classification with SMOTE and anomaly detection techniques | ✅ Complete |
| [Extractive & Abstractive Text Summarization](./text-summarization/) | NLP project comparing TextRank and transformer-based summarization | ✅ Complete |
| [A/B Testing Statistical Analysis](./ab-testing-analysis/) | End-to-end A/B test analysis with power analysis, hypothesis testing, and Bayesian approach | ✅ Complete |
| [Neural Style Transfer with PyTorch](./neural-style-transfer/) | Gatys-style neural style transfer using pretrained VGG19 feature maps | ✅ Complete |
| [Graph Neural Networks for Node Classification (Cora)](./graph-neural-networks-cora/) | Node classification on Cora citation network with GCN from scratch, GraphSAGE, GAT | ✅ Complete |
| [Reinforcement Learning with DQN on CartPole](./reinforcement-learning-dqn/) | DQN agent on CartPole-v1 - Q-network, replay buffer, target net, Double DQN comparison | ✅ Complete |
| [Variational Autoencoder on MNIST](./variational-autoencoder-mnist/) | VAE from scratch in PyTorch - vanilla AE baseline, reparameterization, latent traversals | ✅ Complete |
| [Denoising Diffusion Model on MNIST](./diffusion-models-mnist/) | DDPM from scratch in PyTorch - noise schedule, U-Net noise predictor, ancestral sampling | ✅ Complete |
| [Character-Level Transformer Language Model](./char-transformer-shakespeare/) | Decoder-only transformer from scratch in PyTorch on tiny-shakespeare - tokenization, causal self-attention, autoregressive sampling | ✅ Complete |
| [Self-Supervised Contrastive Learning (SimCLR)](./contrastive-learning-simclr/) | SimCLR representation learning in PyTorch on CIFAR-10 - augmentation pipeline, encoder + projection head, NT-Xent loss, linear probe | ✅ Complete |
| [Deep Convolutional GAN (DCGAN) on Fashion-MNIST](./dcgan-fashion-mnist/) | DCGAN from scratch in PyTorch - transposed-conv generator, strided-conv discriminator, adversarial training, latent interpolation | ✅ Complete |
| [Vision Transformer (ViT) from Scratch on CIFAR-10](./vision-transformer-cifar/) | ViT from scratch in PyTorch - patch embedding, class token, multi-head self-attention encoder, attention-map visualization | ✅ Complete |
| [U-Net Semantic Segmentation on Oxford-IIIT Pet](./unet-segmentation-oxford-pets/) | U-Net from scratch in PyTorch - encoder/decoder with skip connections, joint image/mask augmentation, cross-entropy + Dice loss, mean-IoU | ✅ Complete |
| [Neural Machine Translation with Seq2Seq + Attention](./seq2seq-attention-translation/) | Encoder-decoder LSTM with Bahdanau attention from scratch in PyTorch - parallel-corpus pipeline, additive attention, teacher-forced training, beam decoding + BLEU | ✅ Complete |

## Tech Stack

- Python, NumPy, Pandas, Scikit-learn
- TensorFlow / PyTorch
- Matplotlib, Seaborn, Plotly
- Jupyter Notebooks
