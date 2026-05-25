import gradio as gr
import pandas as pd
import random
from pathlib import Path
import os

# Load three CSV files
csv_files = {
    'gemini': 'results_openrouter_gemini_pro_no_prompt_Diagnostic_Suggestions_scored.csv',
    'gpt': 'results_openrouter_gpt5_2_Diagnostic_Suggestions_scored.csv',
    'qwen': 'results_openrouter_qwen3_Diagnostic_Suggestions_scored.csv'
}

# Rating dimensions
rating_dimensions = ['Accuracy', 'Clarity', 'Relevance', 'Completeness', 'Usefulness']
rating_dimensions_cn = ['Accuracy', 'Clarity', 'Relevance', 'Completeness', 'Usefulness']

# Load data
data = {}
for model_name, filename in csv_files.items():
    df = pd.read_csv(filename)
    data[model_name] = df

# Get filename list (assuming all CSV files have the same filename list)
filenames_list = data['gemini']['filename'].tolist()
max_index = len(filenames_list)
current_index = 1
model_order = []  # Track the current display order of models

# Image folder path (parent folder)
image_folder = Path(__file__).parent.parent

# Rating results storage
ratings_data = []
rated_filenames = set()  # Set of already rated filenames

# Load existing rating records
if Path('blind_ratings.csv').exists():
    existing_ratings = pd.read_csv('blind_ratings.csv')
    if 'filename' in existing_ratings.columns:
        rated_filenames = set(existing_ratings['filename'].tolist())
        ratings_data = existing_ratings.to_dict('records')

def get_image_path(index):
    """Get the image path for the specified index"""
    if 1 <= index <= len(filenames_list):
        filename = filenames_list[index - 1]
        image_path = image_folder / filename
        if image_path.exists():
            return str(image_path)
    return None

def find_next_unrated(start_index):
    """Find the next unrated sample starting from the specified index"""
    for i in range(start_index, max_index + 1):
        filename = filenames_list[i - 1]
        if filename not in rated_filenames:
            return i
    return None

def get_responses(index):
    """Get all model responses for the specified index and randomly shuffle their order"""
    global model_order
    
    if index < 1 or index > len(filenames_list):
        return []
    
    filename = filenames_list[index - 1]
    
    responses = {}
    for model_name, df in data.items():
        row = df[df['filename'] == filename]
        if not row.empty:
            responses[model_name] = row['Response'].values[0]
    
    # Randomly shuffle model order
    model_names = list(responses.keys())
    random.shuffle(model_names)
    model_order = model_names
    
    # Return shuffled responses
    return [responses[name] for name in model_names]

def load_sample(index):
    """Load the sample at the specified index"""
    global current_index
    index = int(index)
    
    # Check if this sample has already been rated
    filename = filenames_list[index - 1] if 1 <= index <= len(filenames_list) else None
    if filename and filename in rated_filenames:
        # If already rated, find the next unrated sample
        next_unrated = find_next_unrated(index)
        if next_unrated is not None:
            index = next_unrated
        else:
            # All samples have been rated
            result = [None, "", "", ""]
            result.extend([3.0] * 15)
            result.append(gr.update(interactive=False))
            return tuple(result)
    
    current_index = index
    responses = get_responses(current_index)
    image_path = get_image_path(current_index)
    
    # Return image, three responses, and reset all sliders (1 image + 3 responses × 5 dimensions = 15 sliders)
    result = [
        image_path,  # Image
        responses[0] if len(responses) > 0 else "",
        responses[1] if len(responses) > 1 else "",
        responses[2] if len(responses) > 2 else "",
    ]
    # Reset all sliders to middle value 3.0
    result.extend([3.0] * 15)  # 3 responses × 5 dimensions
    result.append(gr.update(interactive=True))  # Enable submit button
    
    return tuple(result)

def submit_ratings(*ratings):
    """Submit ratings - receives 15 rating parameters (3 responses × 5 dimensions)"""
    global current_index, model_order
    
    # Parse ratings: first 5 are for Response A's 5 dimensions, middle 5 for Response B, last 5 for Response C
    ratings_list = list(ratings)
    
    # Get the filename of the current sample
    current_filename = filenames_list[current_index - 1]
    
    # Record ratings
    rating_record = {
        'sample_index': current_index,
        'filename': current_filename,
    }
    
    # Record 5 dimension ratings for each model
    for i, model_name in enumerate(model_order):
        for j, dimension in enumerate(rating_dimensions):
            rating_record[f'{model_name}_{dimension}'] = ratings_list[i * 5 + j]
    
    # 添加到已评分集合
    rated_filenames.add(current_filename)
    ratings_data.append(rating_record)
    
    # 保存到CSV
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df.to_csv('blind_ratings.csv', index=False)
    
    # Write ratings back to corresponding results_.csv files
    for i, model_name in enumerate(model_order):
        # Read the corresponding model's CSV file
        csv_file = csv_files[model_name]
        df = pd.read_csv(csv_file)
        
        # Find the row with matching filename
        mask = df['filename'] == current_filename
        
        # Update rating dimensions (only update dimensions in rating_dimensions, not 'Provision of Sources')
        for j, dimension in enumerate(rating_dimensions):
            rating_value = ratings_list[i * 5 + j]
            df.loc[mask, dimension] = rating_value
        
        # Save updated CSV
        df.to_csv(csv_file, index=False)
        
        # Also update data in memory
        data[model_name] = df
    
    message = f"✓ Ratings for sample {current_index} saved and written back to results files!"
    
    # 查找下一个未评分的样本
    next_index = find_next_unrated(current_index + 1)
    if next_index is not None:
        responses = get_responses(next_index)
        image_path = get_image_path(next_index)
        current_index = next_index
        
        result = [
            image_path,  # 图片
            responses[0] if len(responses) > 0 else "",
            responses[1] if len(responses) > 1 else "",
            responses[2] if len(responses) > 2 else "",
        ]
        result.extend([3.0] * 15)  # 重置所有滑动条
        result.append(gr.update(value=next_index))
        result.append(message + f" 已加载样本 {next_index}")
        return tuple(result)
    else:
        result = [gr.update()] * 19  # 1个图片 + 3个文本框 + 15个滑动条
        result.append(gr.update())  # 索引框
        result.append(message + " 所有样本已评分完成！")
        return tuple(result)

def skip_sample():
    """Skip current sample"""
    global current_index
    
    # Find next unrated sample
    next_index = find_next_unrated(current_index + 1)
    if next_index is not None:
        responses = get_responses(next_index)
        image_path = get_image_path(next_index)
        current_index = next_index
        
        result = [
            image_path,  # Image
            responses[0] if len(responses) > 0 else "",
            responses[1] if len(responses) > 1 else "",
            responses[2] if len(responses) > 2 else "",
        ]
        result.extend([3.0] * 15)  # Reset all sliders
        result.append(gr.update(value=next_index))
        result.append(f"Skipped, loaded sample {next_index}")
        return tuple(result)
    else:
        result = [gr.update()] * 19  # 1 image + 3 text boxes + 15 sliders
        result.append(gr.update())  # Index box
        result.append("All samples rated!")
        return tuple(result)
# Create Gradio interface
with gr.Blocks(title="Blind Rating System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Model Response Blind Rating System")
    gr.Markdown("Rating Instructions: Please rate each response on 5 dimensions (1-5 points). The order of the three responses is random, please rate independently.")
    
    with gr.Row():
        sample_index = gr.Number(
            label="Sample Index", 
            value=1, 
            minimum=1, 
            maximum=max_index,
            step=1,
            scale=2
        )
        load_btn = gr.Button("Load Sample", variant="primary", scale=1)
        skip_btn = gr.Button("Skip", scale=1)
    
    gr.Markdown("---")
    
    # Display corresponding image
    with gr.Group():
        gr.Markdown("### 🖼️ Corresponding Image")
        image_display = gr.Image(
            label="",
            type="filepath",
            show_label=False,
            height=400
        )
    
    gr.Markdown("---")
    
    # List to store all slider components
    all_sliders = []
    
    # Response A
    with gr.Group():
        gr.Markdown("### 📄 Response A")
        with gr.Row():
            response1 = gr.Textbox(
                label="",
                lines=15,
                max_lines=25,
                show_label=False,
                interactive=False,
                scale=3
            )
            with gr.Column(scale=1):
                gr.Markdown("**Rating Dimensions:**")
                response1_ratings = []
                for i, (dim_en, dim_cn) in enumerate(zip(rating_dimensions, rating_dimensions_cn)):
                    slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label=f"{dim_en}",
                        info="1-5"
                    )
                    response1_ratings.append(slider)
                    all_sliders.append(slider)
    
    gr.Markdown("---")
    
    # Response B
    with gr.Group():
        gr.Markdown("### 📄 Response B")
        with gr.Row():
            response2 = gr.Textbox(
                label="",
                lines=15,
                max_lines=25,
                show_label=False,
                interactive=False,
                scale=3
            )
            with gr.Column(scale=1):
                gr.Markdown("**Rating Dimensions:**")
                response2_ratings = []
                for i, (dim_en, dim_cn) in enumerate(zip(rating_dimensions, rating_dimensions_cn)):
                    slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label=f"{dim_en}",
                        info="1-5"
                    )
                    response2_ratings.append(slider)
                    all_sliders.append(slider)
    
    gr.Markdown("---")
    
    # Response C
    with gr.Group():
        gr.Markdown("### 📄 响应 C")
        with gr.Row():
            response3 = gr.Textbox(
                label="",
                lines=15,
                max_lines=25,
                show_label=False,
                interactive=False,
                scale=3
            )
            with gr.Column(scale=1):
                gr.Markdown("**Rating Dimensions:**")
                response3_ratings = []
                for i, (dim_en, dim_cn) in enumerate(zip(rating_dimensions, rating_dimensions_cn)):
                    slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label=f"{dim_en}",
                        info="1-5"
                    )
                    response3_ratings.append(slider)
                    all_sliders.append(slider)
    
    gr.Markdown("---")
    
    with gr.Row():
        submit_btn = gr.Button("✓ Submit Ratings", variant="primary", size="lg", scale=2)
        status_text = gr.Textbox(label="Status", scale=3, interactive=False)
    
    gr.Markdown("---")
    gr.Markdown(f"**Total Samples:** {max_index} | **Rating Dimensions:** {', '.join(rating_dimensions)} | **Rating Progress:** See `blind_ratings.csv`")
    
    # Prepare input/output lists
    all_responses = [response1, response2, response3]
    all_outputs = [image_display] + all_responses + all_sliders + [sample_index, status_text]
    all_outputs_for_load = [image_display] + all_responses + all_sliders + [submit_btn]
    
    # Event bindings
    load_btn.click(
        fn=load_sample,
        inputs=[sample_index],
        outputs=all_outputs_for_load
    )
    
    submit_btn.click(
        fn=submit_ratings,
        inputs=all_sliders,
        outputs=all_outputs
    )
    
    skip_btn.click(
        fn=skip_sample,
        inputs=[],
        outputs=all_outputs
    )
    
    # Initially load the first unrated sample
    first_unrated = find_next_unrated(1)
    print(f"Debug: Total filenames: {len(filenames_list)}")
    print(f"Debug: Rated filenames count: {len(rated_filenames)}")
    print(f"Debug: First unrated index: {first_unrated}")
    if first_unrated is None:
        first_unrated = 1
    demo.load(
        fn=load_sample,
        inputs=[gr.Number(value=first_unrated, visible=False)],
        outputs=all_outputs_for_load
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", allowed_paths=[str(image_folder)])