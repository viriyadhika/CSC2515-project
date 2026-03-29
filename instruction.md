**Introduction**

Our project performs an empirical study on Self-Supervised Learning from the lens of environmental sound classification dataset the ESC-50 dataset. ESC-50 consists of 2,000 labeled audio clips, each 5 seconds long, evenly distributed across 50 classes (40 samples per class). The relatively small number of labeled samples makes it difficult to train high-capacity models without overfitting.

A standard pipeline is to convert audio signals into time–frequency representations such as Mel spectrograms and apply image-based models. Initial experiments with simple methods such as K-nearest neighbors (KNN) and softmax regression yield performance close to random guessing (around 5%), which is expected given the 50-way classification setting. This indicates that these methods are not sufficient to capture the structure of the data.

Neural network approaches are commonly used for this task. Prior work shows that neural networks (ViT-based) can achieve strong performance on ESC-50 when combined with sufficient data augmentation or pretraining. However, when trained from scratch on ESC-50 alone, performance is limited due to the dataset size.

To address this limitation, the project explores self-supervised learning (SSL) as a way to leverage additional unlabeled data. While this class of pre-training usually only available on the frontier labs due to the enormous amount of compute required to train these models. This project is our attempt to explore using a smaller dataset in order to experience training a medium size model, observing training dynamics and tuning models. The project eventually build up our understanding towards using a DINO v1 inspired model.

We started our exploration using the SSAST model which is a ViT based backbone model that shows excellent performance in sound dataset upon fine-tuning. To isolate the effect of each of the variables we make one modification at a time compared to the original codebase to build towards DINO model.

**Without pre-training**
On the first section we initialized the model randomly to act as a baseline. The model is immediately trained on SSAST dataset. The following parameter are used (**Refer to the code for audio_ast.py**). The lowest eval loss are chosen and we achieve an accuracy of 40% on test set before the loss taper off.

**Experimental setup**
(See the code and determine.)

**MAE pre-training**

(Description about MAE)

Replicating the result by the original paper with the specified hyperparameter. Instead of using 2 millions audio clip as specified in the paper, our experiment make use of a subset of Audioset consisting of 16K data points. Training this on MAE, we observed that further fine tuning. 

One striking observation is that the standard deviation of the Mel-Spectogram is 12 dB vs. 4dB that was used in the paper (Verify this!). Correcting from the wrong standard deviation assumption, we compute and utilize the mean and standard deviation of the dataset. This correction causes training of MAE to stabilize and as a result we obtain.

(Chart to compare before and after preprocessing)

To observe the benefit of pre-training on the unlabelled data, we compare pre-trained model just using ESC-50 data training set itself. We found that training using just ESC-50 dataset leads to severe overfitting, with poor performance after fine-tuning.

(Chart comparison with and without esc50)

**DINO pre-training**

(Description about DINO)

We eventually try the model of interest (DINO model). We start with the implementation of DINO SR that performs augmentation with teacher receiving full pictures, random crops of images (dino_exp3_asymmetric) with a high dimension head. Through (dino_big_preprocess). One crucial metrics to trace is the mean feature std (describe what it is and how it is calculated in teh code) We suspect that the aggressive nature of the augmentation and high correlation in sound data cause the representation to collapse, showing entropy of just 0.3. Thus, we opt for a softer augmentation and (dino_exp_1_outdim_1024) and also making teacher temperature higher (dino_exp2_temp007). These 2 doesn't exhibit an obvious collapse in mean feature standard dev showing about 0.6 stdev.

(Chart comparison)

To reduce the amount of noise (fact check this, put actual reason), we use gradient accumulation to artificially work with higher batch size without blowing up memory. dino_r2_exp1_accum is using 64 batch size and 256 head, dino_r2_exp2_accum_1024_t07 is using batch size 64 with 1024 head size and teacher temperature 0.07 and dino_r2_exp3_accum_asymmetric is using batch size 64, out dim = 1024 and using original enhanced augmentation with teacher receiving full and student receiving partial image



**KNN as a proxy for embedding quality**
Doing these experiments require us to go back and forth between pre-training and fine-tuning because representation quality is typically unknown until fine tuning is performed eventually. One interesting way we noticed to track if our model is learning is through performing a 0-shot prediction on the validation set of ESC-50 data. In order to do this, we perform a KNN algorithm directly on the embedding space without any fine-tuning. Through the experiments we found that there's a direct correlation between pre-training data size and embedding quality.

(Insert chart on KNN result for MAE with mae_big, mae_big_preprocess, mae_esc50_preprocess and pretrained and also compare the validation accuracy after 45 epochs of fine-tuning)

**Conclusion and Implication**
From this study, we conclude on this particular dataset, MAE is a scalable way to improve performance when the amount of both labelled data and unlabelled data is limited, with performance steadily increasing as the number of data points increases. Performing similar experiments on other datasets will be a way to reaffirm the generalizability of this approach. DINO on the other hand is more sensitive to hyperparameters. It also only benefits from having a huge number of examples.

**Limitation**
The biggest constraint of this study is the compute resources. As most experiments took 3 - 4 hours on RTX4070 even with only 16,000 AudioSet, it's impractical to run with 2 million clips in the scale of SSAST. It would be interesting to explore different ideas

(Data dictionary)
All of these are folders inside data/runs

dino_big_preprocess:
256 dimensions embedding with temperature = 0.04

dino_exp1_outdim1024
Head 1024 dimension

dino_exp2_temp007
Head 1024 dimension with 0.07 teacher temperature

dino_exp3_asymmetric
This is using the original DINO SR augmentation with 1024 dimension embedding

mae_big
Using the original preprocessing script on the paper

mae_big_preprocess
MAE but with preprocessing using the actual dataset standard deviation

mae_esc50_preprocess
MAE pretrained only using ESC50 dataset

mae_25_pct
MAE pretrained only using 25% of ESC50 dataset

For each of these folders, try to check:
<foldername>/metrics.json - contains the knn performance of every epoch during pre-training and final accuracy on the best validation set after fine tuning.
This is except for scratch and pretrained folder. Because we don't do pre-training, just report the final KNN as the final KNN result

pretrain/<checkpoint-*>/trainer_state.json
This is required to build the training vs. validation loss curve during pretraining

final_finetune/<checkpoint-*>/trainer_state.json
This is to build loss and accuracy curve

To access these files, modify the make_charts.py, make_training_curves.py and parse_logs.py to include all the necessary data from these training data. Don't just read the data, but write a python script and execute them to generate data and charts

These are the audio_ast python file that are relevant
- audio_ast.py - The code for pretrained model and scratch
- audio_ast_dino.py - The code for DINO
- audio_ast_mae.py - The code for MAE

Traverse the codebase only for these files because other files are not relevant

These are some of the resources, please cite where it makes sense to cite but at the very least:
SSAST Huggingface - https://huggingface.co/Simon-Kotchou/ssast-small-patch-audioset-16-16 
SSAST paper - https://arxiv.org/abs/2110.09784 
MAE Paper
DINO Paper
DINO SR Paper
ViT Paper

