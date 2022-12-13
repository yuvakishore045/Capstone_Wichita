from happytransformer import HappyTextToText
from happytransformer import TTSettings
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

class pipeline(audio):
    def GEC(self,transcriptions):
        file=open('text.txt','r')
        text=file.read()
        happy_tt = HappyTextToText("T5", "t5-base",load_path='/Model_gec')
        beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=20)
        result = happy_tt.generate_text(text, args=beam_settings)
        file.close()
        return result

    def ASR(self audio):
        model = AutoModelForCTC.from_pretrained("Checkpoints2/checkpoint-107000",local_files_only=True)
        processor = Wav2Vec2Processor.from_pretrained("Checkpoints2/checkpoint-107000",local_files_only=True)
        speech, rate = librosa.load(audio,sr=16000)
        input_values = processor(speech, return_tensors = 'pt').input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim =-1)
        transcriptions = processor.decode(predicted_ids[0])
        file=open('text.txt','w')
        file.write(transcriptions)
        file.close()

#from transformers import pipeline





# import gradio as gr

# def transcribe(audio):
#     text = pipeline(audio)["text"]
#     return text

# gr.Interface(
#     fn=transcribe, 
#     inputs=gr.Audio(source="microphone", type="filepath"), 
#     outputs="text").launch()

# gr.Interface(
#     fn=transcribe, 
#     inputs=[
#         gr.Audio(source="microphone", type="filepath", streaming=True), 
#         "state" 
#     ],
#     outputs=[
#         "textbox",
#         "state"
#     ],
#     live=True).launch()
