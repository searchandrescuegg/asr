from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
import librosa
import nemo.collections.asr as nemo_asr
import gradio as gr
import sys
import torch

try:
    device = torch.device("cuda")
    torch.cuda.get_device_name(0)  # This will fail if no GPU
    print(f"\n\n\nUsing GPU: {torch.cuda.get_device_name(0)}\n\n\n")
except:
    print("No GPU available. Terminating application.")
    sys.exit(1)


# Load the ASR model once at startup
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

app = FastAPI()


@app.post("/api/v1/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file
    """
    try:
        audio_content = await file.read()
        audio_buffer = BytesIO(audio_content)
        audio_array, _ = librosa.load(
            audio_buffer, sr=16000
        )  # NeMo typically expects 16kHz

        if audio_array.ndim > 1:
            raise HTTPException(
                status_code=400, detail="Audio file must be mono channel."
            )

        try:
            transcription = asr_model.transcribe([audio_array])[0]
        except Exception as ex:
            HTTPException(status_code=400, detail=f"Transcription failed: {str(ex)}")

        return {
            "transcription": transcription.text,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


# Gradio interface function
def transcribe_with_gradio(audio_file):
    try:
        if audio_file is None:
            return "No audio file provided"

        # Load audio file
        audio_array, _ = librosa.load(audio_file, sr=16000)

        # Transcribe
        try:
            transcription = asr_model.transcribe([audio_array])[0]
        except Exception as ex:
            return f"Transcription failed: {str(ex)}"

        return transcription.text

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
io = gr.Interface(
    fn=transcribe_with_gradio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Textbox(label="Transcription"),
    title="nvidia/parakeet-tdt-0.6b-v2 Audio Transcription",
    description="Upload an audio file to get its transcription",
)

# Mount Gradio app
app = gr.mount_gradio_app(app, io, path="")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
