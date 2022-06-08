from pydub import AudioSegment
from pydub.utils import make_chunks
import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def run(filepath, extension, chunk_length=1000):

    logging.info("Loading Audio File...")
    myaudio = AudioSegment.from_file(filepath, extension)
    chunks = make_chunks(myaudio, chunk_length)  # Make chunks of one sec

    # Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        logging.info("i/{} Exporting {}...".format(len(chunks), chunk_name))
        chunk.export(chunk_name, format=extension)
