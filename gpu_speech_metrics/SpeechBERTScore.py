# https://github.com/kohei0209/DiscreteSpeechMetrics/blob/mhubert_wo_pypesq/discrete_speech_metrics/speechbertscore.py
import logging
import torch
from transformers import AutoModel
from gpu_speech_metrics.base import BaseMetric

# In PyTorch 2+, a warning for checkpoint mismatch is raised.
# But it should be a false alarm according to the following issue.
# https://github.com/huggingface/transformers/issues/26796
# I have added the following line to suppress the warning.
logging.getLogger("transformers").setLevel(logging.ERROR)

def bert_score(v_generated, v_reference):
    """
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """
    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0))

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean().item()
    recall = torch.max(sim_matrix, dim=0)[0].mean().item()

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


class SpeechBERTScore(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        super().__init__(sample_rate, device)
        # some warnings may appear depending on the environment but should be fine given the discussion below
        # https://huggingface.co/utter-project/mHuBERT-147/discussions/7
        self.model = AutoModel.from_pretrained('utter-project/mHuBERT-147')
        self.model.eval()
        self.model.to(self.device)

    def process_feats(self, audio):
        feats_hiddens = self.model(audio, output_hidden_states=True).hidden_states
        feats = feats_hiddens[8]
        return feats  
    
    def compute_metric(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: Precision.
            float: Recall.
            float: F1 score.
        """
        gt_wav = clean_speech
        gen_wav = denoised_speech
        
        v_ref = self.process_feats(gt_wav)
        v_gen = self.process_feats(gen_wav)

        results = []
        for r, g in zip(v_ref, v_gen):
            precision, recall, f1_score = bert_score(g, r)
            results.append({"SpeechBERTScore": f1_score})

        return results

