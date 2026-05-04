"""cslr_model — Adaptive Sign-to-Gloss training & inference package."""
from .dataset  import Vocabulary, CSLRDataset, build_dataloader, FEAT_DIM, normalize_landmarks
from .model    import AdaptiveBiLSTM, build_model
from .decoder  import CTCPrefixBeamDecoder, CTCBeamDecoder, refine_with_slm
from .metrics  import word_error_rate, character_error_rate, ConfusionMatrix, edit_distance
from .trainer  import CSLRTrainer, EarlyStopping
from .predict  import CSLRPredictor
from .export   import export_torchscript, export_onnx, export_from_checkpoint

__all__ = [
    # data
    "Vocabulary", "CSLRDataset", "build_dataloader", "FEAT_DIM", "normalize_landmarks",
    # model
    "AdaptiveBiLSTM", "build_model",
    # decoding
    "CTCPrefixBeamDecoder", "CTCBeamDecoder", "refine_with_slm",
    # metrics
    "word_error_rate", "character_error_rate", "ConfusionMatrix", "edit_distance",
    # training
    "CSLRTrainer", "EarlyStopping",
    # inference
    "CSLRPredictor",
    # export
    "export_torchscript", "export_onnx", "export_from_checkpoint",
]
