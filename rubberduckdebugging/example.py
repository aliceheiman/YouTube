import os
import speech_recognition as sr
from openai import OpenAI
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch")


def generate_audio(text):
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    return "outputs/output.wav"


def generate_response(text):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert programmer in the form of a rubber ducky to help me debug my code. Please be brief but specific giving guidance on how to resolve my errors. You must respond in one paragraph (no formatting or inline code examples) with a maximum of 50 words.",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return completion.choices[0].message.content


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Quack!")
    audio = r.listen(source)

text = r.recognize_whisper(audio, language="english")
print("The Duck Heard:", text)
print("[🐤] Thinking...")
response = generate_response(text)
print(f"[🐤] {response}")

# Generate audio response
output_audio_path = generate_audio(response)
audio_response = AudioSegment.from_wav(output_audio_path)
play(audio_response)
