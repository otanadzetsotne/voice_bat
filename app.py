import argparse
import os
import tempfile

import whisper
import speech_recognition as sr


def load_model():
    """
    Loads the Whisper model (size "base") to run on CPU.
    """
    print("Loading Whisper model (this may take some time)...")
    model = whisper.load_model("base", device="cpu", in_memory=True)
    return model


def transcribe_audio(model, audio_file):
    """
    Transcribes an audio file using the Whisper model.
    Returns the recognized text.
    """
    print(f"Transcribing audio file: {audio_file}")
    result = model.transcribe(audio_file)
    return result["text"]


def record_audio_chunk(recognizer, source):
    """
    Records a single audio chunk from the microphone until a pause in speech.
    Returns the audio object obtained from the microphone.
    """
    print("Waiting for speech... (Ctrl+C to exit)")
    audio = recognizer.listen(source, phrase_time_limit=30)
    return audio


def listen_mode(output_text_file, model):
    """
    "Listen" mode: continuously records audio from the microphone,
    transcribes each fragment and appends the text to a file.
    Exit by pressing Ctrl+C.
    """
    recognizer = sr.Recognizer()
    # Open the microphone once for continuous listening
    with sr.Microphone() as source:
        print("Starting listening mode. Speak, and the result will be saved to a file.")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                # Record the next audio fragment
                audio = record_audio_chunk(recognizer, source)
                # Save audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio.write(audio.get_wav_data())
                    temp_audio_file = temp_audio.name
                # Transcribe the audio
                transcription = transcribe_audio(model, temp_audio_file)
                # Append the result to the specified text file
                with open(output_text_file, "a", encoding="utf-8") as f:
                    f.write(transcription + "\n")
                print(f"Recognized: {transcription}")
                # Delete the temporary file
                os.remove(temp_audio_file)
        except KeyboardInterrupt:
            print("\nInterruption received. Exiting listening mode.")


def convert_mode(input_audio_file, output_text_file, model):
    """
    "Convert" mode: reads an audio file, transcribes it
    and saves the result to the specified text file.
    """
    transcription = transcribe_audio(model, input_audio_file)
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription result saved to file: {output_text_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Script for converting audio to text with 'listen' and 'convert' modes."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # Subparser for "listen" mode: recording from microphone with continuous transcription
    listen_parser = subparsers.add_parser("listen", help="Continuous recording from microphone and speech transcription")
    listen_parser.add_argument("output_text", type=str, help="Path to output text file")

    # Subparser for "convert" mode: converting an audio file to text
    convert_parser = subparsers.add_parser("convert", help="Converts an audio file to text")
    convert_parser.add_argument("input_audio", type=str, help="Path to input audio file")
    convert_parser.add_argument("output_text", type=str, help="Path to output text file")

    args = parser.parse_args()

    # Load the model once for both modes
    model = load_model()

    if args.mode == "listen":
        listen_mode(args.output_text, model)
    elif args.mode == "convert":
        convert_mode(args.input_audio, args.output_text, model)


if __name__ == '__main__':
    main()
