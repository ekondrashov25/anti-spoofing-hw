import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):
    """
    Class for ASVspoof dataset. Parse protocol and load corresponding audio files.
    """

    def __init__(
        self, protocol_path, audio_dir, target_sample_rate=16000, *args, **kwargs
    ):
        self.target_sample_rate = target_sample_rate
        index = self._create_index(protocol_path, audio_dir)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, audio_dir):
        index = []

        with open(protocol_path, "r") as file:
            for line in file:
                parts = line.strip().split()

                file_name, label_string = parts[1], parts[4]

                label = 1 if label_string == "bonafide" else 0

                index.append(
                    {
                        "path": f"{audio_dir}/{file_name}.flac",
                        "label": label,
                        "utt_id": file_name,
                    }
                )

        return index

    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        return waveform
