import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from Generator.utils import createChunks

load_dotenv(override=True)

instruction = (
    "You are a professional audiobook narrator. "
    "Read the following story in a warm, expressive tone, "
    "with natural pacing, emotional inflection, and pauses after sentences:\n\n"
)


def convert(Args, content, title):

    if "OPEN_AI_KEY" not in os.environ:
        raise Exception("Load OPEN AI Key Access token first !")

    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

    chunks = createChunks(content, limit=Args.Generator.OpenAI.ChunkLimit)

    part_count = 0
    audio_files = []
    for part, chunk in enumerate(chunks):

        mp3file = f"{title}_{part_count}.mp3"

        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions=instruction,
            speed=1.3,
            input=chunk
        )

        with open(mp3file, "wb") as f:
            f.write(response.read())

        audio_files.append(mp3file)
        part_count += 1
        print(f"Part {part_count}/{len(chunks)} done")
        time.sleep(30)




