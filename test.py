from orpheus_tts import OrpheusModel
import wave
import time

model = OrpheusModel(model_name ="canopylabs/3b-es_it-ft-research_release")#, max_model_len=2048)
prompt = '''Amico, è pazzesco come i social media abbiano, ehm, completamente cambiato il modo in cui interagiamo, vero? Siamo tutti connessi 24 ore su 24, 7 giorni su 7, eppure in qualche modo le persone si sentono più sole che mai. E non parliamo nemmeno di quanto stiano incasinando l'autostima e la salute mentale dei ragazzi e tutto il resto.'''

start_time = time.monotonic()
syn_tokens = model.generate_speech(
   prompt=prompt,
   voice="tara",
   )

with wave.open("output.wav", "wb") as wf:
   wf.setnchannels(1)
   wf.setsampwidth(2)
   wf.setframerate(24000)

   total_frames = 0
   chunk_counter = 0
   for audio_chunk in syn_tokens: # output streaming
      chunk_counter += 1
      frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
      total_frames += frame_count
      wf.writeframes(audio_chunk)
   duration = total_frames / wf.getframerate()

   end_time = time.monotonic()
   print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")
