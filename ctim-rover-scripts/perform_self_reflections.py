import argparse
import json
import os
import re
import shutil

SYSTEM_PROMPT="""You are an advanced reasoning agent with staff senior level expertise in software engineering in Python. 
Below follows a past trajectory of an AI agent where it failed solve the reported issue. It may have failed to locate 
all locations in which a fix is required, or produced an incorrect patch for fixing the bug. 

Your task is to reflect on the presented trajectory and distill why the agent was unable to locate or fix the bug and 
what it should do different next time to succeed. You must then provide actionable, insightful information that 
is going to guide the AI agent towards successfully locating all relevant locations that require a fix and generating 
an applicable patch that successfully fixes the bug. Keep your feedback concise, do not exceed more than a few sentences.

Key requirement you MUST follow in any case:
Do not generate an implementation. Do not write code. Do not generate Python. Just provide few sentences in natural language to help the agent.
"""

USER_PROMPT_TEMPLATE="""<FAILED_ATTEMPT>
{trajectory}
</FAILED_ATTEMPT>"""

def perform_self_reflection(trajectory: str, client):
    return client.chat(
        chat=Chat() \
                .add_system(SYSTEM_PROMPT)
                .add_user(USER_PROMPT_TEMPLATE.format(trajectory=trajectory)),
        profile=Profile.ANTHROPIC_CLAUDE_35_SONNET,
        parameters={
            LLMParameters.Temperature: Parameters.FloatValue(0),
        }
    )

def perform_self_reflections_and_cleanup_output(cleanup_mode: bool):
    path_to_output = os.path.join(os.getcwd(), 'output')
    path_to_run_output = os.path.join(os.getcwd(), 'run_output')
    os.makedirs(path_to_run_output, exist_ok=True)
    
    path_to_status = os.path.join(path_to_output, 'current_status_of_all_instances.json')
    path_to_self_reflections = os.path.join(path_to_output, 'self_reflections.json')

    # Load or initialize current status
    if os.path.exists(path_to_status):
        with open(path_to_status, 'r') as f:
            current_status_of_all_instances = json.load(f)

    if not current_status_of_all_instances:
        raise RuntimeError('Could not load current status of all instances. Unable to determine for which instances to '
                           'perform self-reflection without this. Aborting!')

    client = ... # TODO add your LLM client here

    unresolved_instances = [instance_id for instance_id, is_resolved in current_status_of_all_instances.items() if not is_resolved]
    
    # Load or initialize self reflections
    if os.path.exists(path_to_self_reflections):
        with open(path_to_self_reflections, 'r') as f:
            self_reflections = json.load(f)
    else:
        self_reflections = {i: [] for i in unresolved_instances}

    attempt_dirs = [d for d in os.listdir(path_to_run_output) if
                    os.path.isdir(os.path.join(path_to_run_output, d)) and d.startswith('attempt_')]

    if attempt_dirs:
        attempt_indices = [int(d.split('_')[1]) for d in attempt_dirs if d.split('_')[1].isdigit()]
        new_attempt_index = max(attempt_indices) + 1 if attempt_indices else 1
    else:
        new_attempt_index = 1

    new_attempt_dir = os.path.join(path_to_run_output, f'attempt_{new_attempt_index}')
    resolved_dir = os.path.join(path_to_run_output, f'resolved_instances')
    os.makedirs(resolved_dir, exist_ok=True)
    os.makedirs(new_attempt_dir, exist_ok=True)

    completed_self_reflections = 0
    total_self_reflections = len(unresolved_instances)
    for dirpath, dirnames, filenames in os.walk(path_to_output):
        if not dirnames and filenames:
            instance_id = dirpath.split('/')[-1].split('_2025')[0]
            if instance_id in unresolved_instances and not cleanup_mode:
                completed_self_reflections += 1
                print(f'Performing self-reflection for {instance_id}. {completed_self_reflections}/{total_self_reflections}')
                available_trajectory_indices = [int(match.group(1)) for file in filenames if (match := re.search(rf'debug_agent_write_patch_(\d).json', file))]
                if available_trajectory_indices:
                    max_trajectory_index = max(available_trajectory_indices)
                    with open(os.path.join(dirpath, f'debug_agent_write_patch_{max_trajectory_index}.json'), 'r') as f:
                        trajectory = json.load(f)

                        # Remove exemplar, dont need to include this in the reflection, will just introduce noise
                        if len(trajectory) > 1 and 'Below follows an example' in trajectory[1]['content']:
                            del trajectory[1]

                        formatted_trajectory = ''
                        for message in trajectory:
                            formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"

                        response = perform_self_reflection(formatted_trajectory, client)
                        self_reflections[instance_id].append(response.content)

                        print(f'Reflection: {response.content}\n')

                        cost_file = os.path.join(dirpath, 'cost.json')
                        try:
                            with open(cost_file, 'r') as f:
                                current_cost = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            current_cost = {"total_cost": 0}
                        
                        current_cost['total_cost'] += float(response.spent.amount) / 100000
                        
                        with open(cost_file, 'w') as f:
                            json.dump(current_cost, f)
                else:
                    self_reflections[instance_id].append('No patch-generation trajectory found. Perhaps there was some problem with the generation of the patch.')

            # Calculate relative path from path_to_output to preserve directory structure
            rel_path = os.path.relpath(dirpath, path_to_output)

            # Create the same directory structure in new_attempt_dir
            if instance_id in unresolved_instances:
                dst_dir = os.path.join(new_attempt_dir, rel_path)
            else:
                dst_dir = os.path.join(resolved_dir, rel_path)
            os.makedirs(dst_dir, exist_ok=True)

            for filename in filenames:
                if filename not in ['self_reflections.json', 'current_status_of_all_instances.json']:
                    src_path = os.path.join(dirpath, filename)
                    dst_path = os.path.join(dst_dir, filename)
                    shutil.move(src_path, dst_path)

            # Ensure that the output sub-folders are clean for the next run
            os.removedirs(dirpath)

    with open(path_to_self_reflections, 'w+') as f:
        json.dump(self_reflections, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('--cleanup-mode', action='store_true',
                        help="If set, skips self reflections and only performs cleanup duties.")

    args = parser.parse_args()

    perform_self_reflections_and_cleanup_output(cleanup_mode=args.cleanup_mode)
