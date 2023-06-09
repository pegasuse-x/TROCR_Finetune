import matplotlib
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import paths, constants

def load_processor() -> TrOCRProcessor:
    return TrOCRProcessor.from_pretrained(paths.trocr_repo)


def load_model(from_disk: bool) -> VisionEncoderDecoderModel:
    if from_disk:
        assert paths.model_path.exists(), f"No model existing at {paths.model_path}"
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.model_path)
        debug_print(f"Loaded local model from {paths.model_path}")
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.trocr_repo)
        debug_print(f"Loaded pretrained model from huggingface ({paths.trocr_repo})")

    debug_print(f"Using device {constants.device}.")
    model.to(constants.device)
    return model


def init_model_for_training(model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4


def debug_print(string: str):
    if constants.should_log:
        print(string)