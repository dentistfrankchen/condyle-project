import requests
import base64
import json
import os
import csv
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm

def get_image_base64(image_path):
    """Convert local image to base64 encoded string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def call_openrouter_vl(api_key, image_path, prompt, model_name="google/gemini-3-pro-preview", max_retries=3, retry_delay=2):
    """Call OpenRouter API for image recognition with retry mechanism
    
    Args:
        api_key: OpenRouter API key
        image_path: Path to image file
        prompt: Text prompt for the model
        model_name: Model name to use (default: "google/gemini-3-pro-preview")
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Convert image
    base64_image = get_image_base64(image_path)
    
    # Build request payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract result
            content = result['choices'][0]['message']['content']
            return content
            
        except Exception as e:
            print(f"OpenRouter API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Returning None.")
                return None

def parse_result_with_keywords(response_text, keywords):
    """Parse API response to 0 or 1 based on keywords
    
    Args:
        response_text: API response text
        keywords: List of keywords. If response contains any keyword, return 1
    
    Returns:
        0 or 1
    """
    if not response_text:
        return 0
    
    response_lower = response_text.lower()

    print(response_lower)
    
    # Check keywords
    for keyword in keywords:
        if keyword.lower() in response_lower:
            return 1
    
    return 0

def get_processed_files(csv_path):
    """Get list of already processed files from CSV"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            if 'filename' in df.columns:
                return set(df['filename'].tolist())
        except:
            pass
    return set()

def select_images_to_process(folder_path, csv_path):
    """Display images and let user select which ones to process"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    folder = Path(folder_path)
    all_images = sorted([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])
    
    if not all_images:
        print("No images found in folder.")
        return []
    
    processed_files = get_processed_files(csv_path)
    
    print("\n" + "=" * 60)
    print("Available Images:")
    print("=" * 60)
    for idx, img in enumerate(all_images, 1):
        status = "[PROCESSED]" if img.name in processed_files else "[NEW]      "
        print(f"{idx:3d}. {status} {img.name}")
    print("=" * 60)
    
    print("\nSelection options:")
    print("  - Enter 'all' to process all images")
    print("  - Enter 'new' to process only new images")
    print("  - Enter ranges like '1-5' or '1,3,5' or '1-3,5,7-9'")
    print("  - Processed images will be overwritten if selected")
    
    while True:
        user_input = input("\nYour selection: ").strip().lower()
        
        if user_input == 'all':
            return all_images
        elif user_input == 'new':
            return [img for img in all_images if img.name not in processed_files]
        else:
            try:
                selected_indices = set()
                parts = user_input.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.update(range(start, end + 1))
                    else:
                        selected_indices.add(int(part))
                
                selected_images = [all_images[i-1] for i in selected_indices if 1 <= i <= len(all_images)]
                if selected_images:
                    print(f"\nSelected {len(selected_images)} images for processing.")
                    return selected_images
                else:
                    print("No valid images selected. Try again.")
            except:
                print("Invalid input. Try again.")

def create_expert_csv(main_csv_path, questions, shared_expert_csv='expert_annotations.csv'):
    """Create or update shared expert evaluation CSV with only new entries
    
    Args:
        main_csv_path: Path to the model's result CSV
        questions: List of question dictionaries
        shared_expert_csv: Shared expert CSV filename (same for all models)
    """
    # Use shared expert CSV in the same folder as main CSV
    csv_folder = os.path.dirname(main_csv_path)
    expert_csv_path = os.path.join(csv_folder, shared_expert_csv)
    
    # Read main CSV
    if not os.path.exists(main_csv_path):
        print(f"Main CSV not found: {main_csv_path}")
        return
    
    df_main = pd.read_csv(main_csv_path, encoding='utf-8-sig')
    
    # Check if expert CSV exists
    if os.path.exists(expert_csv_path):
        df_expert = pd.read_csv(expert_csv_path, encoding='utf-8-sig')
        existing_files = set(df_expert['filename'].tolist())
        # Get only new entries
        df_new = df_main[~df_main['filename'].isin(existing_files)].copy()
    else:
        df_new = df_main.copy()
    
    if df_new.empty:
        print(f"No new entries to add to expert CSV: {expert_csv_path}")
        return
    
    # Create expert dataframe with only filename and empty columns
    df_expert_new = pd.DataFrame()
    df_expert_new['filename'] = df_new['filename']
    
    # Add all question columns as empty
    for question in questions:
        qname = question['name']
        if question.get('type') == 'binary':
            # For binary: add result column with 'wait' to indicate needs filling
            df_expert_new[qname] = 'wait'
            df_expert_new[f'{qname}_raw'] = ''  # Raw column stays empty
        else:
            # For text: add response column, empty
            df_expert_new[qname] = ''
    
    # Append or create expert CSV
    if os.path.exists(expert_csv_path):
        df_expert_existing = pd.read_csv(expert_csv_path, encoding='utf-8-sig')
        df_expert_combined = pd.concat([df_expert_existing, df_expert_new], ignore_index=True)
        df_expert_combined.to_csv(expert_csv_path, index=False, encoding='utf-8-sig')
        print(f"Appended {len(df_new)} new entries to shared expert CSV: {expert_csv_path}")
    else:
        df_expert_new.to_csv(expert_csv_path, index=False, encoding='utf-8-sig')
        print(f"Created shared expert CSV with {len(df_new)} entries: {expert_csv_path}")

def save_responses_to_json(json_path, filename, question_name, response, api_provider, model_name):
    """Save original API response to JSON file
    
    Args:
        json_path: Path to JSON file
        filename: Image filename
        question_name: Name of the question
        response: API response text
        api_provider: API provider name
        model_name: Model name used
    """
    # Load existing data if file exists
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []
    
    # Create new entry
    entry = {
        'filename': filename,
        'question': question_name,
        'api_provider': api_provider,
        'model': model_name,
        'response': response
    }
    
    # Check if entry already exists and update it
    existing_index = None
    for idx, item in enumerate(data):
        if (item.get('filename') == filename and 
            item.get('question') == question_name and
            item.get('api_provider') == api_provider and
            item.get('model') == model_name):
            existing_index = idx
            break
    
    if existing_index is not None:
        data[existing_index] = entry
    else:
        data.append(entry)
    
    # Save to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_images_with_multiple_questions(folder_path, api_provider, api_key, questions, output_csv, selected_images=None, model_name=None):
    """Process selected images with multiple questions and save results to CSV
    
    Args:
        folder_path: Path to image folder
        api_provider: only supports 'openrouter'
        api_key: API key
        questions: List of question dictionaries:
            - 'name': Question name
            - 'prompt': Question prompt
            - 'type': 'binary' (yes/no) or 'text' (descriptive)
            - 'keywords': Keyword list (only for binary type)
        output_csv: Output CSV file path
        selected_images: List of selected image files to process (if None, user will be prompted)
        model_name: Model name to use (if None, uses default for the provider)
    """
    # If no images selected, prompt user
    if selected_images is None:
        selected_images = select_images_to_process(folder_path, output_csv)
        if not selected_images:
            print("No images selected. Exiting.")
            return
    
    image_files = selected_images
    
    print(f"Found {len(image_files)} images")
    print(f"Using API: {api_provider}")
    print(f"Number of questions: {len(questions)}")
    binary_count = sum(1 for q in questions if q.get('type') == 'binary')
    text_count = sum(1 for q in questions if q.get('type') == 'text')
    print(f"  - Binary questions: {binary_count}")
    print(f"  - Text questions: {text_count}")
    print("-" * 50)
    
    # Select API function (OpenRouter only)
    if api_provider != 'openrouter':
        raise ValueError("api_provider must be 'openrouter'")
    api_function = lambda key, path, prompt: call_openrouter_vl(
        key, path, prompt, model_name=model_name if model_name else "google/gemini-3-pro-preview"
    )
    actual_model_name = model_name if model_name else "google/gemini-3-pro-preview"
    
    # Prepare JSON output path
    json_output = output_csv.replace('.csv', '_responses.json')
    
    # Prepare results storage
    results = []
    # Prepare text question responses for scored files
    text_responses = {q['name']: [] for q in questions if q.get('type') == 'text'}
    
    # Process each image with progress bar
    for image_file in tqdm(image_files, desc=f"Processing with {api_provider}", unit="image"):
        row_result = {'filename': image_file.name}
        
        # Ask each question
        for question in questions:
            question_name = question['name']
            prompt = question['prompt']
            question_type = question.get('type', 'binary')  # Default is binary
            keywords = question.get('keywords', [])
            
            print(f"  Question: {question_name} [{question_type}]")
            
            # Call API
            response = api_function(api_key, str(image_file), prompt)
            
            # Save original response to JSON file
            save_responses_to_json(json_output, image_file.name, question_name, response, api_provider, actual_model_name)
            
            # Process result based on question type
            if question_type == 'binary':
                # Binary question: Parse to 0 or 1, and save original response
                result = parse_result_with_keywords(response, keywords)
                tqdm.write(f"    [{image_file.name}] {question_name}: {result}")
                row_result[f'{question_name}'] = result
                row_result[f'{question_name}_raw'] = response  # Save original response
            else:
                # Text question: Save full response
                tqdm.write(f"    [{image_file.name}] {question_name}: {response[:50] if response else 'None'}...")
                row_result[f'{question_name}'] = response
                # Collect for scored file
                text_responses[question_name].append(response)
        
        results.append(row_result)
    
    # Generate header dynamically
    fieldnames = ['filename']
    for question in questions:
        fieldnames.append(question['name'])
        # Add raw response column for binary questions
        if question.get('type', 'binary') == 'binary':
            fieldnames.append(f"{question['name']}_raw")
    
    # Create dataframe from new results
    df_new = pd.DataFrame(results)
    
    # Merge with existing CSV if it exists
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv, encoding='utf-8-sig')
        # Remove rows that are being reprocessed
        processed_filenames = set(df_new['filename'].tolist())
        df_existing = df_existing[~df_existing['filename'].isin(processed_filenames)]
        # Combine
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Updated CSV: {output_csv}")
    else:
        df_new.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Created CSV: {output_csv}")
    
    # Create scored files for text questions
    for question in questions:
        if question.get('type') == 'text':
            question_name = question['name']
            scored_filename = output_csv.replace('.csv', f'_{question_name}_scored.csv')
            
            # Prepare data for scored file - use filename instead of index
            scored_data = []
            for result in results:
                filename = result['filename']
                response = result.get(question_name, '')
                scored_data.append({
                    'filename': filename,
                    'Response': response,
                    'Accuracy': '',
                    'Clarity': '',
                    'Relevance': '',
                    'Completeness': '',
                    'Provision of Sources': '',
                    'Usefulness': ''
                })
            
            # Create or append to scored file using pandas
            df_new = pd.DataFrame(scored_data)
            
            if os.path.exists(scored_filename):
                # Append to existing file
                df_existing = pd.read_csv(scored_filename, encoding='utf-8-sig')
                # Remove rows with same filenames to avoid duplicates
                processed_filenames = set(df_new['filename'].tolist())
                df_existing = df_existing[~df_existing['filename'].isin(processed_filenames)]
                # Append
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(scored_filename, index=False, encoding='utf-8-sig')
                print(f"Appended to scored file: {scored_filename}")
            else:
                # Create new file
                df_new.to_csv(scored_filename, index=False, encoding='utf-8-sig')
                print(f"Created scored file: {scored_filename}")
    
    print("-" * 50)
    print(f"Processing complete! Results saved to: {output_csv}")
    print(f"Original responses saved to: {json_output}")
    print(f"Processed {len(results)} images, {len(questions)} questions")
    
    # Create expert CSV
    create_expert_csv(output_csv, questions)

# --- Usage Example ---
if __name__ == "__main__":
    # Configuration
    OPENROUTER_API_KEY = "<REDACTED_API_KEY>"
    
    IMAGE_FOLDER = r"./"  # Image folder path
    
    # Define multiple questions (supports two types)
    QUESTIONS = [
        {
            'name': 'Diagnostic_Suggestions',
            'type': 'text',
            'prompt': 'You are now a senior expert in oral and maxillofacial radiology. Write a structured CBCT report based on DC/TMD diagnostic criteria.'
        }

    ]
    
    # ====================================================================
    # SELECT IMAGES TO PROCESS (shared for all API providers)
    # ====================================================================
    # Select images once for all providers
    selected_images = select_images_to_process(IMAGE_FOLDER, 'results_openrouter_gemini_pro_no_prompt.csv')
    if not selected_images:
        print("No images selected. Exiting.")
        exit(0)
    
    # ====================================================================
    # PROCESS WITH DIFFERENT API PROVIDERS AND MODELS
    # To add a new model:
    # 1. Add a new process_images_with_multiple_questions() call below
    # 2. Set api_provider to 'openrouter'
    # 3. Set model_name parameter to the desired model
    # 4. Use a unique output_csv filename
    #
    # Available OpenRouter models: "google/gemini-3-pro-preview", 
    #                               "anthropic/claude-3-opus", "openai/gpt-4-vision-preview"
    # ====================================================================
    # Process with OpenRouter API - Gemini Pro Preview
    print("\n" + "=" * 60)
    print("Processing images with OpenRouter API - Gemini 3 Pro Preview")
    print("=" * 60)
    process_images_with_multiple_questions(
        folder_path=IMAGE_FOLDER,
        api_provider='openrouter',
        api_key=OPENROUTER_API_KEY,
        questions=QUESTIONS,
        output_csv='results_openrouter_gemini_pro_no_prompt.csv',
        selected_images=selected_images,
        model_name="google/gemini-3-pro-preview"
    )

    print("\n" + "=" * 60)
    print("Processing images with OpenRouter API - gpt5.2 Max Preview")
    print("=" * 60)
    process_images_with_multiple_questions(
        folder_path=IMAGE_FOLDER,
        api_provider='openrouter',
        api_key=OPENROUTER_API_KEY,
        questions=QUESTIONS,
        output_csv='results_openrouter_gpt5_2.csv',
        selected_images=selected_images,
        model_name="openai/gpt-5.2"
    )

    print("\n" + "=" * 60)
    print("Processing images with OpenRouter API - Qwen 3 Max Preview")
    print("=" * 60)
    process_images_with_multiple_questions(
        folder_path=IMAGE_FOLDER,
        api_provider='openrouter',
        api_key=OPENROUTER_API_KEY,
        questions=QUESTIONS,
        output_csv='results_openrouter_qwen3.csv',
        selected_images=selected_images,
        model_name="qwen/qwen3-vl-235b-a22b-thinking")
    
    # Process with OpenRouter API - Default Model
    '''print("\n" + "=" * 60)
    print("Processing images with OpenRouter API - Gemini 3 Pro Preview")
    print("=" * 60)
    process_images_with_multiple_questions(
        folder_path=IMAGE_FOLDER,
        api_provider='openrouter',
        api_key=OPENROUTER_API_KEY,
        questions=QUESTIONS,
        output_csv='results_openrouter_multi.csv',
        selected_images=selected_images,
        model_name="google/gemini-3-pro-preview"  # Can change model here
    )'''
    

