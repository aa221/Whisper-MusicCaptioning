# Whisper-MusicCaptioning
Leveraging Whisper on the Musiccaps dataset, to accurately caption music.


Audio captioning plays a crucial role in various applications. Media companies dealing with large amounts of unlabeled audio and visual content require automated solutions to efficiently generate accurate captions. Additionally, the growing demand for labeled data in AI development has made audio annotation more essential than ever. However, manually labeling audio segments is both time-consuming and costly.

Automated audio captioning (AAC) is a task that replicates human perception while creatively connecting audio processing with natural language processing. AAC has seen significant advances in recent years. A survey of advances and challenges in AAC can be found in Xu et al. (2024).

Captioning models offer a scalable and cost-effective alternative that enables more efficient data annotation while enhancing accessibility and content discovery. Similar logic applies to music. Distributors and streaming services often require a way to describe objective segments of songs to enhance recommendations and personalization. Currently, these platforms primarily rely on non-textual features, such as tempo, to generate recommendations. However, this approach has limitations. Although useful, these metrics lack the depth needed to fully capture the objective qualities of a song. For example, multiple genres and moods can share the same tempo, making it an insufficient differentiator for nuanced music discovery.

The AudioCaps (Audiocaps, 2019) and Clotho (Clotho, 2020) datasets are both widely used for training audio captioning models. AudioCaps consists of over 50,000 audio samples, each paired with descriptive captions that detail the sounds present, enabling models to learn accurate and contextually relevant descriptions of diverse audio events. The Clotho dataset includes more than 5,000 audio clips, each annotated with multiple human-generated captions that capture various aspects and interpretations of the audio content. Together, these datasets provide comprehensive resources that support the development and evaluation of robust audio captioning systems, facilitating advancements in areas such as accessibility, multimedia content organization, and human-computer interaction. Although these datasets contain some music, they are not focused on this task, as they primarily include natural sounds and other environmental events.

Along these lines, audio captioning models perform quite well. Models like SLAM-AAC (Chen et al., 2024) achieve a METEOR score of 0.268. On the other hand, music captioning models are far from mature. The best-performing model, LP-MusicCaps (Doh et al., 2023), achieved a METEOR score of 0.22.

Our aim is to propose a lightweight, robust solution to music captioning that can be leveraged in industry settings. We propose fine-tuning OpenAI's speech-to-text model on the MusicCaps dataset (Agostinelli et al., 2023) so that it can accurately caption music. In addition, we create an interface to showcase the use of our model, allowing larger corporations to integrate it into their systems. Lastly, we introduce a novel approach to evaluating music captioning—leveraging subjective and objective word frequencies, as well as a large language model (LLM).

Our key contribution in this work is a music captioning model with an easy-to-use interface. Users can upload or drag-and-drop melodies to generate textual captions. Our work can be accessed and replicated through the provided resources.


## Whisper-MusicCaptioning
A breakdown of this repository. 

1. Download_local.ipynb demonstrates how to download the youtube videos that exist within the Musiccaps dataset. This is crucial if you would like to replicate the finetuning process on this dataset.
2. finetune.py is the straightforward training script used to finetune Whisper on the musiccaps dataset.
3. musiccaps-public.csv is the musiccaps dataset. We have included in this repo for your convenience.
4. Expierements.ipynb is our subjective, and objective term frequency analysis. It explains how well the model does describing subjective and objective terms of a given piece of music.


Please refer to the paper for more in depth derivations. Also note, our weights are publicly available on huggingface: https://huggingface.co/Adel2214/musiccaps-whisper

You can also interact with the model via our demo here: https://captionmymusic.vercel.app/home.



## Future goals
The results in the paper show that there is some merit in leveraging pre-trained models to do music captioning. While the subjective term analysis has highlighted that our model struggles to accurately identify the correct subjective terms of a song, we believe that with more data, music captioning models will eventually close this gap. We hope to further expirement with different methods to improve the subjective analysis of music—including potentially leveraging LLMs to maximize this metric across generated captions.
