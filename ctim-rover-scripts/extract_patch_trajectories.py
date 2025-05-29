import ast
import sys
from pymilvus import MilvusClient
import os
import json
import re

def extract_patch_trajectories():
    """
    Traverse the resolved_instances/applicable_patch folders and extract the highest-indexed
    debug_agent_write_patch JSON files along with their corresponding instance IDs.
    Also collects failed attempts for each successfully resolved instance.
    
    Returns:
        list: A list of dictionaries, each containing 'instance_id', 'patch_generation_trajectory',
              and 'failed_attempts'
    """
    # Path to resolved instances with applicable patches
    base_path = os.path.join("run_output", "resolved_instances", "applicable_patch")
    run_output_path = "run_output"
    
    # List to store the results
    trajectories = []
    
    # Walk through all directories in the resolved_instances/applicable_patch folder
    for dirpath, dirnames, filenames in os.walk(base_path):
        # Filter for relevant JSON files
        patch_files = [f for f in filenames if re.match(r'debug_agent_write_patch_\d+\.json', f)]
        
        if patch_files:
            # Extract the indices from the filenames
            indices = [int(re.search(r'debug_agent_write_patch_(\d+)\.json', f).group(1)) for f in patch_files]
            
            # Find the file with the highest index
            max_index = max(indices)
            max_patch_file = f'debug_agent_write_patch_{max_index}.json'
            
            # Extract instance_id from the parent folder name
            dirname = os.path.basename(dirpath)
            instance_id = dirname.split('_2025')[0]
            
            # Load the JSON file
            file_path = os.path.join(dirpath, max_patch_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                patch_data = json.load(f)
            
            # Add to our list of trajectories (without failed attempts for now)
            trajectories.append({
                'instance_id': instance_id,
                'patch_generation_trajectory': patch_data,
                'source': 'swe_bench_verified',
                'failed_attempts': []  # Initialize empty list for failed attempts
            })
    
    # Create a dictionary for fast lookup of trajectories by instance_id
    trajectory_map = {t['instance_id']: t for t in trajectories}
    
    # Collect failed attempts by traversing attempt folders only once
    for i in range(1, 4):  # Limit to 3 attempts
        attempt_dir = os.path.join(run_output_path, f"attempt_{i}")
        if not os.path.exists(attempt_dir):
            continue
            
        # Walk through subdirectories in this attempt
        for dirpath, dirnames, filenames in os.walk(attempt_dir):
            # Only process leaf directories (has files but no subdirectories)
            # And ensure it follows the expected naming pattern with timestamp
            if filenames and not dirnames and '_2025' in os.path.basename(dirpath):
                # Extract instance_id from the directory name using the same pattern
                dirname = os.path.basename(dirpath)
                instance_id = dirname.split('_2025')[0]
                
                # Check if this instance_id exists in our successful trajectories
                if instance_id in trajectory_map:
                    # Found a directory for a successful instance, look for patch files
                    patch_files = [f for f in filenames if re.match(r'debug_agent_write_patch_\d+\.json', f)]
                    if patch_files:
                        # Extract the indices from the filenames
                        indices = [int(re.search(r'debug_agent_write_patch_(\d+)\.json', f).group(1)) for f in patch_files]
                        
                        # Find the file with the highest index
                        if len(indices) > 0:
                            max_index = max(indices)
                            max_patch_file = f'debug_agent_write_patch_{max_index}.json'
                            
                            # Load the JSON file
                            file_path = os.path.join(dirpath, max_patch_file)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                patch_data = json.load(f)
                            
                            # Add to failed attempts for the matching trajectory
                            trajectory_map[instance_id]['failed_attempts'].append({
                                'attempt_number': i,
                                'patch_data': patch_data
                            })
    
    return trajectories

if __name__ == "__main__":
    # Extract the trajectories
    patch_trajectories = extract_patch_trajectories()
    
    # Print summary
    print(f"Found {len(patch_trajectories)} trajectory files")
    
    milvus_client = MilvusClient('data/task_embeddings.db')
    all_lite_ids = [i for i in range(0, 53)]
    swe_bench_lite_trajectories = milvus_client.get(collection_name='swe_bench_lite', ids=all_lite_ids)
    
    for t in swe_bench_lite_trajectories:
        t['source'] = 'swe_bench_lite'
        t['patch_generation_trajectory'] = ast.literal_eval(t['patch_generation_trajectory'])
    
    combined_trajectories = [
        {
            'instance_id': t['instance_id'],
            'patch_generation_trajectory': t['patch_generation_trajectory'],
            'source': t['source'],
            'repository': t['instance_id'].split('_')[0],
            'failed_attempts': t.get('failed_attempts', [])  # Include failed attempts in the output
        }
        for t in swe_bench_lite_trajectories + patch_trajectories
    ]

    # Save combined trajectories
    output_file = "data/all_successful_patch_trajectories.json"
    with open(output_file, 'w') as f:
        json.dump(combined_trajectories, f, indent=2)

    print(f"Saved {len(combined_trajectories)} combined trajectories to {output_file}")