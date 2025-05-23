import os
from together import Together
import json
from tqdm import tqdm
import argparse

whisper_transcripts = json.load(open("/data1/soumyad/IS2025_challenge/valid_new_dict.json", "r"))

transcripts = dict(zip(whisper_transcripts["path"], whisper_transcripts["text"]))


client = Together(api_key=<together api key>)
output = {}
os.makedirs("whisper_llama70b", exist_ok=True)

for i, (k, v) in enumerate(tqdm(transcripts.items())):
    output_file = os.path.join("whisper_llama70b", k.split(os.sep)[-1].replace(".wav", ".txt"))
    if os.path.exists(output_file):
        continue
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo",
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        messages=[
                    {"role": "system", "content": "You are an emotion recognizer capable of understanding the emotion within a text. (Your task) Classify the following sentence into one of the 8 emotions: happy, angry, sad, neutral, contempt, disgust, surprise, or fear. Reply with only 1 word."},
                {"role": "user", "content": f"sentence: {v}"}
                ]
    )
    output_text = response.choices[0].message.content
    with open(output_file, "w") as f:
        f.write(output_text)
        f.close()
