# https://github.com/kohei0209/DiscreteSpeechMetrics/blob/mhubert_wo_pypesq/discrete_speech_metrics/speechbertscore.py
import logging
import torch
from transformers import AutoModel
from gpu_speech_metrics.base import BaseMetric

# Enable TF32 for better performance
torch.set_float32_matmul_precision("high")

# In PyTorch 2+, a warning for checkpoint mismatch is raised.
# But it should be a false alarm according to the following issue.
# https://github.com/huggingface/transformers/issues/26796
# Suppress transformer warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


class SpeechBERTScore(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        # some warnings may appear depending on the environment but should be fine given the discussion below
        # https://huggingface.co/utter-project/mHuBERT-147/discussions/7
        self.model = AutoModel.from_pretrained("utter-project/mHuBERT-147")
        self.model.to(self.device)
        self.model.eval()

        try:
            self.model = torch.compile(self.model, mode="default")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")

    @staticmethod
    def f1_score(denoised_embedding: torch.Tensor, clean_embedding: torch.Tensor) -> float:
        similarity_matrix = torch.matmul(denoised_embedding, clean_embedding.T) / (
            torch.norm(denoised_embedding, dim=1, keepdim=True) * torch.norm(clean_embedding, dim=1).unsqueeze(0)
        )

        precision = torch.max(similarity_matrix, dim=1)[0].mean().item()
        recall = torch.max(similarity_matrix, dim=0)[0].mean().item()

        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    def get_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                feats_hiddens = self.model(audio, output_hidden_states=True).hidden_states
                feats = feats_hiddens[8]
                return feats

    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        assert clean_speech is not None
        clean_embeddings = self.get_embeddings(clean_speech)
        denoised_embeddings = self.get_embeddings(denoised_speech)

        results = []
        for clean_embedding, denoised_embedding in zip(clean_embeddings, denoised_embeddings, strict=False):
            f1_score = self.f1_score(denoised_embedding, clean_embedding)
            results.append({"SpeechBERTScore": f1_score})

        return results
