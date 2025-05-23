# ABHINAYA

This is the repository for our submission to the **Speech Emotion Recognition in Naturalistic Conditions Challenge** held at **Interspeech 2025**. We ranked **4th** out of **166** teams in the categorical task. However, one of our models were not fully trained within the challenge deadline. Upon completion, our system ranked **1st** in the challenge.

The leaderboard is available at [https://lab-msp.com/MSP-Podcast_Competition/IS2025/]

## System Description

Our system consisted of an ensemble of 5 different classifiers

* S1: A speech only model - WavLM - trained with weighted focal loss
* S2: A speech LLM - SALMONN13B - trained with weighted focal loss
* T1: A zero-shot text-only LLM - LLaMA 3.3 70B
* T2: A fine-tuned text-only LLM - LLaMA 3.1 8B - trained with vector scaling (VS) loss
* ST1: A speech LLM adapted for speech-text joint modeling - SALMONN 7B - trained with vector scaling (VS) loss

Finally, a majority voting was taken among the outputs of all the classifiers which resulted in **44.02%** macro-F1 score

### System S1

* Run **audio_model.py** for this system

### System S2

* We fine-tune the **SALMONN-13B** model for this task. Please refer to [https://github.com/bytedance/SALMONN] for understanding and settung up this code.
* You just have to download the SALMONN-13B file, the BEATS tokenizer and change the corresponding paths for running the **train_podcast.py** file.
* The test results will be automatically saved as a csv file.

### System T1

* For this go into the **T1** folder, change the path of the files in the **test_dict.json** file to your own test files.
* Run **get_whisper_asr.py** to get the transcripts for the test files.
* Get an account in the Together platform [https://www.together.ai/models/llama-3-3-70b] (or run it locally) and get the labels for the test files.

### System T2

* The transcripts for the train and balanced validation sets are already provided: **train_new_dict.json** and **valid_new_dict.json**.
* The test transcripts from the **Step 2** of **System T1** are to be used for the inference part of the code.
* This part requires CUDA 12 or higher to run




