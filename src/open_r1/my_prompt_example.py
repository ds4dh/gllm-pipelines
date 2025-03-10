import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


def get_model(
    model_path_or_id: str,
    saved_as_peft: bool=False,
):
    """ Load a PEFT model from the huggingface hub or from a local path
    """
    # Select model loading function
    if saved_as_peft:
        model = AutoPeftModelForCausalLM.from_pretrained
    else:
        model_loading_fn = AutoModelForCausalLM.from_pretrained
    
    # Load model
    model = model_loading_fn(
        model_path_or_id,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Merge PEFT model for vLLM if required
    if saved_as_peft:
        model = model.merge_and_unload()
        # model.save_pretrained(merged_model_path)
    
    return model


def main():
    # Load PEFT model
    model = get_model(cfg.peft_model_path_or_id)
        
    # Load samples from CSV
    df = pd.read_csv("your_samples.csv")
    prompts = df['prompt_column'].tolist()

    # Set up vLLM inference
    llm = LLM(model="merged_model_path")  # vLLM loads HuggingFace models directly
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    # Fast batch inference
    outputs = llm.generate(prompts, sampling_params)

    # Save responses back to CSV
    df['responses'] = [output.outputs[0].text for output in outputs]
    df.to_csv("your_samples_with_responses.csv", index=False)
