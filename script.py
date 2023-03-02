from pytube import YouTube
import tempfile
import asyncio
import whisper
import os

async def audio_transcriber(path):
    model = whisper.load_model("base")
    audio = whisper.load_audio(path)

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model.device)
    _, probs = model.detect_language(mel)

    result = model.transcribe(audio=audio, language=max(probs, key=probs.get), fp16=False)
    return result['text']

async def mp3_converter():
    print('Enter your youtube video URL:')
    ytURl = input()
    
    if not 'www.youtube.com' in ytURl: 
       return print('Invalid URL')

    with tempfile.TemporaryDirectory() as path:
        YouTube(ytURl).streams.get_lowest_resolution().download(path, 'audio.mp3')
        print('Wait for the transcription process...')
        audio_text = await audio_transcriber(os.path.join(path, 'audio.mp3'))

        return print(audio_text)

asyncio.run(mp3_converter())
