import json
import os
import re

def update_current_status_and_remaining_tasks():
    #'data', 'runs_stored', 'benchmark_on_test_NO_rules_INCL_exemplars_from_o1_with_gpt_4o_and_combined_lite_and_verified_data'
    path_to_output = os.path.join(os.getcwd(), 'data', 'runs_stored', 'benchmark_on_test_REPO_rules_NO_exemplars_from_o1_with_gpt_4o_and_combined_lite_and_verified_data')
    path_to_task_list = os.path.join(os.getcwd(), 'conf', 'remaining_tasks.txt')
    path_to_status = os.path.join(path_to_output, 'current_status_of_all_instances.json')

    # Initialize status_list from existing file if available
    if os.path.exists(path_to_status):
        with open(path_to_status, 'r') as f:
            current_status = json.load(f)
    else:
        current_status = {}
        
    for dirpath, dirnames, filenames in os.walk(path_to_output):
        if not dirnames and filenames:
            instance_id = dirpath.split('/')[-1].split('_2025')[0]
            if 'applicable_patch' in dirpath: # are in an instance's result folder
                available_trajectory_indices = [int(match.group(1)) for file in filenames if (match := re.search(rf'debug_agent_write_patch_(\d).json', file))]
                if not available_trajectory_indices:
                    current_status[instance_id] = False
                else:
                    max_trajectory_index = max(available_trajectory_indices)
                    with open(os.path.join(dirpath, f'debug_agent_write_patch_{max_trajectory_index}.json'), 'r') as f:
                        trajectory = json.load(f)
                        current_status[instance_id] = 'resolves the issue. Congratulations!' in trajectory[-1]['content']
            else:
                # If we couldnt apply a patch, we know it must have been false for sure.
                # In this case we cant even run tests.
                current_status[instance_id] = False

    unresolved_instances = [instance_id for instance_id, is_resolved in current_status.items() if not is_resolved]

    with open(path_to_task_list, 'w') as f:
        f.write('\n'.join(unresolved_instances))

    with open(path_to_status, 'w+') as f:
        json.dump(current_status, f)

    costs = 0
    for dirpath, dirnames, filenames in os.walk(path_to_output):
        if 'cost.json' in filenames:
            with open(os.path.join(dirpath, 'cost.json'), 'r') as f:
                cost_file = json.load(f)
                costs += cost_file['total_cost']

    runs = len(current_status)
    
    with open(os.path.join(path_to_output, 'costs.json'), 'w') as f:
        json.dump({'total_cost': costs, 'cost_per_sample': costs/runs}, f)
    print(f'Evaluating on SWE-Bench Verified test split with 3 retries cost: {costs}$ total or {costs/runs}$ per sample.')

if __name__ == "__main__":
    update_current_status_and_remaining_tasks()
