# Note that after training, you can easily load the model 
# using the .from_pretrained(output_dir) method.

# Also on another note this inference is only for testing the new model 
# which trained on the new dataset. You can use this as a basic 
# example to integrate the new model on to your webapp. 

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader


# need a lot of work
# expected to be done after the training issue resolved


class TrocrPredictor:
    def __init__(self, use_local_model: bool = True):
        self.processor = load_processor()
        self.model = load_model(use_local_model)

    def predict_for_image_paths(self, image_paths: list[str]) -> list[tuple[str, float]]:
        images = [Image.open(path) for path in image_paths]
        return self.predict_images(images)

    def predict_images(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        dataset = MemoryDataset(images, self.processor)
        dataloader = DataLoader(dataset, constants.batch_size)
        predictions, confidence_scores = predict(self.processor, self.model, dataloader)
        return zip([p[1] for p in sorted(predictions)], [p[1] for p in sorted(confidence_scores)])

def predict(
    processor: TrOCRProcessor, model: VisionEncoderDecoderModel, dataloader: DataLoader
) -> tuple[list[tuple[int, str]], list[float]]:
    output: list[tuple[int, str]] = []
    confidence_scores: list[tuple[int, float]] = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            debug_print(f"Predicting batch {i+1}")
            inputs: torch.Tensor = batch["input"].to(constants.device)

            generated_ids = model.generate(inputs, return_dict_in_generate=True, output_scores = True)
            generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            batch_confidence_scores = get_confidence_scores(generated_ids)
            confidence_scores.extend(zip(ids, batch_confidence_scores))

    return output, confidence_scores


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

def debug_print(string: str):
    if constants.should_log:
        print(string)

# load images
image_names = ["data/img1.png", "data/img2.png"]
images = [Image.open(img_name) for img_name in image_names]

# directly predict on Pillow Images or on file names
model = TrocrPredictor()
predictions = model.predict_images(images)
predictions = model.predict_for_file_names(image_names)

# print results
for i, file_name in enumerate(image_names):
    print(f'Prediction for {file_name}: {predictions[i]}')