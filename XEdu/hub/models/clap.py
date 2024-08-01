import onnxruntime
import soundfile as sf
import numpy as np
class CLAP:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)


    def read_audio(self, audio_path):
        audio_time_series, sample_rate = sf.read(audio_path)
        resample_rate = 44100
        return audio_time_series.astype(np.float32), resample_rate

    def load_audio_into_tensor(self,audio_path, audio_duration):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = self.read_audio(audio_path)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        # return torch.FloatTensor(audio_time_series)
        return audio_time_series 

    def preprocess_audio(self,audio_files):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        duration = 7
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(audio_file, duration)
            audio_tensor = audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return audio_tensors
    
    def get_audio_embedding(self,audio_files):
        r"""Compute audio embeddings for a list of audio files"""
        preprocessed_audio = self.preprocess_audio(audio_files)
        ort_session = self.model
        res = []
        for i in preprocessed_audio:
            ort_inputs = {ort_session.get_inputs()[0].name: i}
            
            ort_outs = ort_session.run(None, ort_inputs)[0]
            res.append(ort_outs[0])
        return np.stack(res)
    