from pathlib import Path
import shutil
import csv
import gc
import zipfile
import tempfile
import time
import resource

import gradio as gr
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


# =========================
# Settings
# =========================

MODEL_NAME = "facebook/dinov2-large"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def increase_open_file_limit(target=8192):
    """
    Large Gradio folder uploads can open many temporary files at once.
    Raising the soft limit helps avoid "Too many open files" on macOS/Linux.
    """
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft_limit = min(target, hard_limit)

        if soft_limit < new_soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
            print(f"Open file limit increased: {soft_limit} -> {new_soft_limit}")
    except (OSError, ValueError) as error:
        print(f"Could not increase open file limit: {error}")


increase_open_file_limit()


# =========================
# Device
# =========================

if torch.backends.mps.is_available():
    device = "mps"      # Apple Silicon Mac
elif torch.cuda.is_available():
    device = "cuda"     # NVIDIA GPU
else:
    device = "cpu"

print("Using device:", device)


# =========================
# Load DINOv2 once
# =========================

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# =========================
# Helper functions
# =========================

def safe_copy(src_path, dst_dir):
    """
    Copy file into dst_dir while preserving the original filename.
    """
    src_path = Path(src_path)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / src_path.name

    if dst_path.exists():
        raise gr.Error(
            "Duplicate output filename found: "
            f"{src_path.name}. Rename one source file or split the run."
        )

    shutil.copy2(src_path, dst_path)
    return dst_path


def make_zip(folder_path, zip_path):
    """
    Zip all files inside a folder.
    """
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(folder_path))

    return zip_path


def make_comparison_image(kept_path, removed_path, similarity, output_path):
    """
    Create a side-by-side preview image showing the kept reference and removed image.
    """
    kept_image = load_image(kept_path)
    removed_image = load_image(removed_path)

    if kept_image is None or removed_image is None:
        return None

    tile_size = (420, 420)
    label_height = 64
    padding = 16

    kept_image.thumbnail(tile_size, Image.Resampling.LANCZOS)
    removed_image.thumbnail(tile_size, Image.Resampling.LANCZOS)

    canvas_width = tile_size[0] * 2 + padding * 3
    canvas_height = tile_size[1] + label_height + padding * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    kept_x = padding
    removed_x = tile_size[0] + padding * 2
    image_y = label_height + padding

    canvas.paste(
        kept_image,
        (kept_x + (tile_size[0] - kept_image.width) // 2, image_y)
    )
    canvas.paste(
        removed_image,
        (removed_x + (tile_size[0] - removed_image.width) // 2, image_y)
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((kept_x, padding), "KEPT", fill="green")
    draw.text((kept_x, padding + 22), Path(kept_path).name[:48], fill="black")
    draw.text((removed_x, padding), "REMOVED - SIMILAR", fill="red")
    draw.text((removed_x, padding + 22), Path(removed_path).name[:48], fill="black")
    draw.text(
        (removed_x, padding + 42),
        f"Similarity: {similarity:.4f}",
        fill="black"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def load_image(path):
    """
    Safely load an image as RGB.
    """
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def resolve_folder_path(local_folder_path):
    """
    Resolve and validate a local folder path from user input.
    """
    folder_path = Path(local_folder_path.strip().strip("\"'")).expanduser()

    if not folder_path.exists():
        raise gr.Error(f"Local folder does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise gr.Error(f"Local path is not a folder: {folder_path}")

    return folder_path


def collect_images_in_folder(folder_path):
    """
    Collect image paths inside one folder (including nested files).
    """
    return sorted(
        path
        for path in Path(folder_path).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_image_paths(uploaded_files, local_folder_path):
    """
    Collect image paths from either Gradio uploads or a local folder path.
    Reading from a local path avoids Gradio upload limits for large batches.
    """
    if local_folder_path and local_folder_path.strip():
        return collect_images_in_folder(resolve_folder_path(local_folder_path))

    if not uploaded_files:
        return []

    return [
        Path(file)
        for file in uploaded_files
        if Path(file).suffix.lower() in IMAGE_EXTENSIONS
    ]


def discover_subfolders(parent_path):
    """
    Find immediate child directories that contain images.
    Returns (subfolders_with_images, skipped_folder_names).
    """
    parent = resolve_folder_path(parent_path).resolve()
    subfolders = []
    skipped = []

    for child in sorted(parent.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        if collect_images_in_folder(child):
            subfolders.append(child)
        else:
            skipped.append(child.name)

    return subfolders, skipped


def get_output_name_prefix(local_folder_path):
    """
    Use the selected local folder name as the ZIP filename prefix.
    """
    if local_folder_path and local_folder_path.strip():
        return resolve_folder_path(local_folder_path).name

    return "images"


def cleanup_between_folders():
    """
    Release memory between independent subfolder runs.
    """
    gc.collect()

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def extract_embeddings(image_paths, batch_size, progress_callback=None):
    """
    Extract DINOv2 embeddings for all images.
    Batch size only controls how many images are processed at once.
    All embeddings are compared later across the whole dataset.
    """
    all_embeddings = []
    valid_paths = []

    total_batches = max(1, (len(image_paths) + batch_size - 1) // batch_size)

    for batch_num, i in enumerate(range(0, len(image_paths), batch_size)):
        if progress_callback:
            progress_callback(
                batch_num / total_batches,
                f"Extracting DINOv2 embeddings: batch {batch_num + 1}/{total_batches}"
            )

        batch_paths = image_paths[i:i + batch_size]
        images = []
        paths = []

        for path in batch_paths:
            image = load_image(path)
            if image is not None:
                images.append(image)
                paths.append(path)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token = one feature vector for the whole image
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Normalize embeddings so dot product becomes cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        valid_paths.extend(paths)

    if not all_embeddings:
        return None, []

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, valid_paths


# =========================
# Main DINOv2 similarity function
# =========================

EMPTY_RESULT = (
    "No images found. Upload images or enter a local folder path.",
    None,
    None,
    None,
    [],
    [],
    []
)


def process_folder(
    image_paths,
    folder_name,
    output_dir,
    similarity_threshold,
    similarity_threshold_percent,
    batch_size,
    preview_limit,
    progress,
    progress_start,
    progress_end,
):
    """
    Run deduplication for one folder and write ZIP/CSV outputs.
    """
    def report(fraction, desc):
        progress(
            progress_start + (progress_end - progress_start) * fraction,
            desc=desc
        )

    result = {
        "folder_name": folder_name,
        "error": None,
        "valid_count": 0,
        "kept_count": 0,
        "duplicate_count": 0,
        "unique_zip_path": None,
        "duplicate_zip_path": None,
        "report_path": None,
        "comparison_gallery_items": [],
        "unique_gallery_items": [],
        "duplicate_gallery_items": [],
    }

    if not image_paths:
        result["error"] = "No valid image files found."
        return result

    work_root = Path(tempfile.mkdtemp()) / folder_name
    unique_dir = work_root / "unique_images"
    duplicate_dir = work_root / "similar_images"
    comparison_dir = work_root / "similarity_comparisons"

    unique_dir.mkdir(parents=True, exist_ok=True)
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_zip_path = output_dir / f"{folder_name}_unique.zip"
    duplicate_zip_path = output_dir / f"{folder_name}_similar.zip"
    report_path = output_dir / f"{folder_name}_report.csv"

    try:
        report(0.05, f"{folder_name}: preparing images")

        def embedding_progress(batch_fraction, desc):
            report(0.05 + batch_fraction * 0.65, f"{folder_name}: {desc}")

        embeddings, valid_paths = extract_embeddings(
            image_paths=image_paths,
            batch_size=int(batch_size),
            progress_callback=embedding_progress
        )

        if embeddings is None or len(valid_paths) == 0:
            result["error"] = "Could not read any images."
            return result

        result["valid_count"] = len(valid_paths)

        kept_indices = []
        duplicates = []

        report(0.75, f"{folder_name}: comparing image similarity")

        for idx in range(len(valid_paths)):
            current_embedding = embeddings[idx]

            if len(kept_indices) == 0:
                kept_indices.append(idx)
                continue

            kept_embeddings = embeddings[kept_indices]
            similarities = torch.matmul(kept_embeddings, current_embedding)
            max_similarity, max_position = torch.max(similarities, dim=0)
            max_similarity = float(max_similarity.item())
            most_similar_kept_index = kept_indices[max_position.item()]

            if max_similarity >= similarity_threshold:
                duplicates.append({
                    "duplicate_index": idx,
                    "duplicate_image": valid_paths[idx],
                    "kept_index": most_similar_kept_index,
                    "kept_image": valid_paths[most_similar_kept_index],
                    "similarity": max_similarity
                })
            else:
                kept_indices.append(idx)

        del embeddings

        report(0.85, f"{folder_name}: copying files to output folders")

        for idx in kept_indices:
            safe_copy(valid_paths[idx], unique_dir)

        for item in duplicates:
            safe_copy(item["duplicate_image"], duplicate_dir)

        with open(report_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["similar_image", "kept_reference_image", "similarity"])

            for item in duplicates:
                writer.writerow([
                    Path(item["duplicate_image"]).name,
                    Path(item["kept_image"]).name,
                    round(item["similarity"], 6)
                ])

        report(0.95, f"{folder_name}: creating zip files")

        make_zip(unique_dir, unique_zip_path)
        make_zip(duplicate_dir, duplicate_zip_path)

        comparison_gallery_items = []
        for count, item in enumerate(duplicates[:int(preview_limit)]):
            comparison_path = make_comparison_image(
                kept_path=item["kept_image"],
                removed_path=item["duplicate_image"],
                similarity=item["similarity"],
                output_path=comparison_dir / f"comparison_{count:05d}.jpg"
            )

            if comparison_path is not None:
                caption = (
                    f"{folder_name} | Kept: {Path(item['kept_image']).name} | "
                    f"Removed: {Path(item['duplicate_image']).name} | "
                    f"Similarity: {item['similarity']:.4f}"
                )
                comparison_gallery_items.append((str(comparison_path), caption))

        unique_gallery_items = []
        for idx in kept_indices[:int(preview_limit)]:
            image_path = valid_paths[idx]
            caption = f"{folder_name} | Unique: {Path(image_path).name}"
            unique_gallery_items.append((str(image_path), caption))

        duplicate_gallery_items = []
        for item in duplicates[:int(preview_limit)]:
            caption = (
                f"{folder_name} | Similar to: {Path(item['kept_image']).name} | "
                f"Similarity: {item['similarity']:.4f}"
            )
            duplicate_gallery_items.append((str(item["duplicate_image"]), caption))

        result.update({
            "valid_count": len(valid_paths),
            "kept_count": len(kept_indices),
            "duplicate_count": len(duplicates),
            "unique_zip_path": str(unique_zip_path),
            "duplicate_zip_path": str(duplicate_zip_path),
            "report_path": str(report_path),
            "comparison_gallery_items": comparison_gallery_items,
            "unique_gallery_items": unique_gallery_items,
            "duplicate_gallery_items": duplicate_gallery_items,
        })

        report(1.0, f"{folder_name}: done")
        return result

    except Exception as error:
        result["error"] = str(error)
        return result

    finally:
        if "valid_paths" in locals():
            del valid_paths
        shutil.rmtree(work_root, ignore_errors=True)
        cleanup_between_folders()


def process_mass_folders(
    parent_path,
    similarity_threshold_percent,
    batch_size,
    preview_limit,
    progress=gr.Progress(),
):
    """
    Process each immediate subfolder independently.
    """
    parent = resolve_folder_path(parent_path).resolve()
    subfolders, skipped = discover_subfolders(parent_path)
    similarity_threshold = similarity_threshold_percent / 100

    if len(subfolders) < 2:
        return None

    total = len(subfolders)
    results = []
    failed = []

    for index, subfolder in enumerate(subfolders):
        slice_start = index / total
        slice_end = (index + 1) / total

        progress(slice_start, desc=f"Processing {subfolder.name} ({index + 1}/{total})")

        folder_result = process_folder(
            image_paths=collect_images_in_folder(subfolder),
            folder_name=subfolder.name,
            output_dir=parent,
            similarity_threshold=similarity_threshold,
            similarity_threshold_percent=similarity_threshold_percent,
            batch_size=batch_size,
            preview_limit=preview_limit,
            progress=progress,
            progress_start=slice_start,
            progress_end=slice_end,
        )

        if folder_result["error"]:
            failed.append((subfolder.name, folder_result["error"]))
        else:
            results.append(folder_result)

    progress(1.0, desc="Mass processing complete")

    if not results and failed:
        error_lines = "\n".join(f"- {name}: {message}" for name, message in failed)
        return (
            f"Mass processing failed for all {total} folders.\n\n{error_lines}",
            None,
            None,
            None,
            [],
            [],
            []
        )

    summary_lines = [
        "Mass subfolder processing complete.",
        "",
        f"Parent folder: {parent}",
        f"Model: {MODEL_NAME}",
        f"Device: {device}",
        f"Similarity threshold: {similarity_threshold_percent}% ({similarity_threshold:.2f})",
        f"Folders processed: {len(results)} / {total}",
        "",
        "Per-folder results:",
    ]

    for folder_result in results:
        summary_lines.append(
            f"- {folder_result['folder_name']}: "
            f"{folder_result['valid_count']} images, "
            f"{folder_result['kept_count']} kept, "
            f"{folder_result['duplicate_count']} similar"
        )
        summary_lines.append(f"  {folder_result['unique_zip_path']}")
        summary_lines.append(f"  {folder_result['duplicate_zip_path']}")

    if skipped:
        summary_lines.extend(["", "Skipped folders with no images:"])
        summary_lines.extend(f"- {name}" for name in skipped)

    if failed:
        summary_lines.extend(["", "Failed folders:"])
        summary_lines.extend(f"- {name}: {message}" for name, message in failed)

    preview_result = results[0]
    summary_lines.extend([
        "",
        f"Gallery previews show results from: {preview_result['folder_name']}",
    ])

    return (
        "\n".join(summary_lines),
        [result["unique_zip_path"] for result in results],
        [result["duplicate_zip_path"] for result in results],
        [result["report_path"] for result in results],
        preview_result["comparison_gallery_items"],
        preview_result["unique_gallery_items"],
        preview_result["duplicate_gallery_items"],
    )


def remove_similar_images(
    uploaded_files,
    local_folder_path,
    similarity_threshold_percent,
    batch_size,
    preview_limit,
    progress=gr.Progress()
):
    """
    uploaded_files: images uploaded through Gradio
    local_folder_path: local folder path for large image batches
    similarity_threshold_percent: 50 to 99 from the slider
    batch_size: how many images DINOv2 processes at once
    preview_limit: how many unique/similar images to show in each gallery
    """
    if local_folder_path and local_folder_path.strip():
        subfolders, _ = discover_subfolders(local_folder_path)
        if len(subfolders) >= 2:
            return process_mass_folders(
                parent_path=local_folder_path,
                similarity_threshold_percent=similarity_threshold_percent,
                batch_size=batch_size,
                preview_limit=preview_limit,
                progress=progress,
            )

    image_paths = collect_image_paths(uploaded_files, local_folder_path)

    if not image_paths:
        return EMPTY_RESULT

    similarity_threshold = similarity_threshold_percent / 100

    if local_folder_path and local_folder_path.strip():
        output_dir = Path(tempfile.mkdtemp()) / f"dinov2_dedupe_{int(time.time())}"
        folder_name = get_output_name_prefix(local_folder_path)
    else:
        output_dir = Path(tempfile.mkdtemp()) / f"dinov2_dedupe_{int(time.time())}"
        folder_name = "images"

    folder_result = process_folder(
        image_paths=image_paths,
        folder_name=folder_name,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        similarity_threshold_percent=similarity_threshold_percent,
        batch_size=batch_size,
        preview_limit=preview_limit,
        progress=progress,
        progress_start=0.0,
        progress_end=1.0,
    )

    if folder_result["error"]:
        if folder_result["error"] == "No valid image files found.":
            return (
                "No valid image files found. Use JPG, JPEG, PNG, BMP, or WEBP.",
                None,
                None,
                None,
                [],
                [],
                []
            )

        raise gr.Error(folder_result["error"])

    summary = (
        f"DINOv2 similarity cleanup complete.\n\n"
        f"Model: {MODEL_NAME}\n"
        f"Device: {device}\n"
        f"Similarity threshold: {similarity_threshold_percent}% "
        f"({similarity_threshold:.2f})\n\n"
        f"Uploaded valid images: {folder_result['valid_count']}\n"
        f"Unique images kept: {folder_result['kept_count']}\n"
        f"Similar images found: {folder_result['duplicate_count']}\n\n"
        f"Download {folder_name}_unique.zip and {folder_name}_similar.zip below."
    )

    progress(1.0, desc="Done")

    return (
        summary,
        folder_result["unique_zip_path"],
        folder_result["duplicate_zip_path"],
        folder_result["report_path"],
        folder_result["comparison_gallery_items"],
        folder_result["unique_gallery_items"],
        folder_result["duplicate_gallery_items"],
    )


# =========================
# Gradio UI
# =========================

with gr.Blocks(title="DINOv2 Image Similarity Remover") as demo:
    gr.Markdown(
        """
        # DINOv2 Image Similarity Remover

        For large image sets, enter a local folder path below instead of using browser upload.
        Browser upload can fail when thousands of temporary files are opened at once.

        **Mass processing:** Enter a parent folder with multiple subfolders (e.g. `Class 1 Frames`).
        Each subfolder is processed independently and outputs `{subfolder}_unique.zip` and
        `{subfolder}_similar.zip` into the parent folder.

        This app uses **Hugging Face DINOv2** embeddings and **cosine similarity**
        to separate unique images from very similar images.

        **Important:** This does not delete your original files.  
        It creates downloadable ZIP files.
        """
    )

    local_folder_path = gr.Textbox(
        label="Local folder path for large or mass batches",
        placeholder="/Users/richard/Pictures/Class 1 Frames",
        info="Single folder: processes that folder. Parent folder with 2+ image subfolders: mass mode — each subfolder processed independently."
    )

    uploaded_files = gr.File(
        label="Upload image folder for smaller batches",
        file_count="directory",
        file_types=["image"],
        type="filepath"
    )

    with gr.Row():
        similarity_threshold = gr.Slider(
            minimum=50,
            maximum=99,
            value=95,
            step=1,
            label="Similarity threshold (%)"
        )

        batch_size = gr.Slider(
            minimum=1,
            maximum=64,
            value=16,
            step=1,
            label="Batch size"
        )

        preview_limit = gr.Slider(
            minimum=5,
            maximum=100,
            value=30,
            step=5,
            label="Preview limit for each gallery"
        )

    run_button = gr.Button("Run DINOv2 Similarity Cleanup")

    summary_output = gr.Textbox(
        label="Summary",
        lines=10
    )

    with gr.Row():
        unique_zip_output = gr.File(
            label="Download unique images (one or many ZIPs in mass mode)",
            file_count="multiple"
        )
        duplicate_zip_output = gr.File(
            label="Download similar images (one or many ZIPs in mass mode)",
            file_count="multiple"
        )
        report_csv_output = gr.File(
            label="Download CSV report(s)",
            file_count="multiple"
        )

    comparison_gallery = gr.Gallery(
        label="Similarity report: kept image vs removed image",
        columns=1,
        height=600
    )

    unique_gallery = gr.Gallery(
        label="Preview of unique/kept images",
        columns=4,
        height=500
    )

    duplicate_gallery = gr.Gallery(
        label="Preview of similar/removed images",
        columns=4,
        height=500
    )

    run_button.click(
        fn=remove_similar_images,
        inputs=[
            uploaded_files,
            local_folder_path,
            similarity_threshold,
            batch_size,
            preview_limit
        ],
        outputs=[
            summary_output,
            unique_zip_output,
            duplicate_zip_output,
            report_csv_output,
            comparison_gallery,
            unique_gallery,
            duplicate_gallery
        ]
    )


demo.queue(default_concurrency_limit=1).launch(
    max_file_size="4GB",
    show_error=True,
    allowed_paths=[str(Path.home())],
)