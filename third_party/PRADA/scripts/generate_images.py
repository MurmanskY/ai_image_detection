from pathlib import Path

import fire

from prada.wrappers import get_wrapper


def main(
    model: str,
    output_dir: str | Path,
    batch_size: int = 1,
    samples_per_class: int = 10,
    prompts_csv_path: str | None = None,
    prompts_csv_name_col: str = "image name (matching Raise-1k)",
    prompts_csv_prompt_col: str = "Prompt",
    seed: int = 0,
):
    """Generate images with a selected wrapper and save them to output_dir.

    If prompts_csv_path is provided, generation uses prompts from the CSV (using prompts from prompts_csv_prompt_col and
    filenames according to prompts_csv_name_col). Otherwise, samples samples_per_class images per ImageNet class.
    """

    # create output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        if input(f"Path '{output_dir}' already exists, continue? (y/n) ").lower() not in ["yes", "y"]:
            exit()
    output_dir.mkdir(parents=True, exist_ok=True)

    # load wrapper
    wrapper = get_wrapper(model=model)

    if prompts_csv_path is not None:
        wrapper.sample_from_prompts(
            output_dir=output_dir,
            prompts_csv_path=prompts_csv_path,
            name_col=prompts_csv_name_col,
            prompt_col=prompts_csv_prompt_col,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        wrapper.sample_imagenet(
            output_dir=output_dir, samples_per_class=samples_per_class, batch_size=batch_size, seed=seed
        )


if __name__ == "__main__":
    fire.Fire(main)
